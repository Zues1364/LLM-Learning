from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import pickle
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import psutil
from huggingface_hub import model_info
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from crawler import crawl_url  # noqa: E402
from utils import FAISSVectorStore, normalize_for_match  # noqa: E402


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    query_prefix: str = ""
    doc_prefix: str = ""
    trust_remote_code: bool = False


MODEL_SPECS: Dict[str, ModelSpec] = {
    "aiteamvn": ModelSpec(
        key="aiteamvn",
        hf_id="AITeamVN/Vietnamese_Embedding",
    ),
    "gte-multilingual-base": ModelSpec(
        key="gte-multilingual-base",
        hf_id="Alibaba-NLP/gte-multilingual-base",
        trust_remote_code=True,
    ),
    "multilingual-e5-small": ModelSpec(
        key="multilingual-e5-small",
        hf_id="intfloat/multilingual-e5-small",
        query_prefix="query: ",
        doc_prefix="passage: ",
    ),
    "bkai-vietnamese-bi-encoder": ModelSpec(
        key="bkai-vietnamese-bi-encoder",
        hf_id="bkai-foundation-models/vietnamese-bi-encoder",
    ),
    "dangvantuan-vietnamese-embedding": ModelSpec(
        key="dangvantuan-vietnamese-embedding",
        hf_id="dangvantuan/vietnamese-embedding",
    ),
}


class MemorySampler:
    def __init__(self, interval_sec: float = 0.2):
        self.interval_sec = interval_sec
        self.process = psutil.Process(os.getpid())
        self._peak_rss = self.process.memory_info().rss
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            try:
                rss = self.process.memory_info().rss
                if rss > self._peak_rss:
                    self._peak_rss = rss
            except Exception:
                pass
            time.sleep(self.interval_sec)

    def stop(self) -> float:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        return float(self._peak_rss)


class BenchmarkEmbedder:
    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self.model_name = spec.hf_id
        self.model = SentenceTransformer(
            spec.hf_id,
            trust_remote_code=spec.trust_remote_code,
        )
        self.embedding_dim = int(self.model.get_sentence_embedding_dimension())

    def _prep_doc(self, text: str) -> str:
        return f"{self.spec.doc_prefix}{text}" if self.spec.doc_prefix else text

    def _prep_query(self, text: str) -> str:
        return f"{self.spec.query_prefix}{text}" if self.spec.query_prefix else text

    def embed_documents_np(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        prepared = [self._prep_doc(text) for text in texts]
        embeddings = self.model.encode(
            prepared,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(embeddings, dtype="float32")

    def embed_query(self, text: str) -> List[float]:
        prepared = self._prep_query(text)
        embedding = self.model.encode(
            [prepared],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )[0]
        return np.asarray(embedding, dtype="float32").tolist()


def _read_cases(cases_path: Path) -> Dict[str, Any]:
    return json.loads(cases_path.read_text(encoding="utf-8"))


def _load_pdf_cache(path: Path) -> List[Document]:
    docs = pickle.loads(path.read_bytes())
    if not isinstance(docs, list):
        raise ValueError(f"Unexpected cache format: {path}")
    return docs


def _apply_resource_filters(docs: List[Document], resource: Dict[str, Any]) -> List[Document]:
    filters = resource.get("filters") or {}
    if not filters:
        return docs

    page_in = {int(v) for v in filters.get("page_in", [])}
    contains_any = [normalize_for_match(str(v)) for v in filters.get("contains_any", []) if str(v).strip()]

    filtered: List[Document] = []
    for doc in docs:
        if page_in:
            try:
                page_value = int(doc.metadata.get("page") or 0)
            except Exception:
                page_value = 0
            if page_value not in page_in:
                continue

        if contains_any:
            haystack = normalize_for_match(doc.page_content)
            if not any(token in haystack for token in contains_any):
                continue

        filtered.append(doc)

    return filtered


def _load_html_docs(path: Path, file_id: str) -> List[Document]:
    docs = crawl_url(str(path))
    for idx, doc in enumerate(docs):
        meta = dict(doc.metadata or {})
        meta["file_id"] = file_id
        meta["file_name"] = path.name
        meta["chunk_index"] = int(meta.get("chunk_index", idx))
        meta["index"] = int(meta.get("index", idx + 1))
        doc.metadata = meta
    return docs


def load_corpus(repo_root: Path, config: Dict[str, Any]) -> tuple[List[Document], Dict[str, Dict[str, Any]]]:
    docs: List[Document] = []
    resources_by_id: Dict[str, Dict[str, Any]] = {}
    for resource in config["resources"]:
        resource_id = resource["id"]
        full_path = repo_root / resource["path"]
        if resource["kind"] == "pdf_cache":
            loaded = _load_pdf_cache(full_path)
        elif resource["kind"] == "html_local":
            loaded = _load_html_docs(full_path, resource["file_id"])
        else:
            raise ValueError(f"Unsupported resource kind: {resource['kind']}")

        loaded = _apply_resource_filters(loaded, resource)

        for doc in loaded:
            meta = dict(doc.metadata or {})
            meta["benchmark_resource_id"] = resource_id
            meta["benchmark_label"] = resource["label"]
            meta.setdefault("file_id", resource["file_id"])
            meta.setdefault("file_name", Path(resource["file_id"]).name)
            doc.metadata = meta
        docs.extend(loaded)
        resources_by_id[resource_id] = resource
    return docs, resources_by_id


def corpus_signature(docs: Iterable[Document]) -> str:
    digest = hashlib.sha256()
    for doc in docs:
        digest.update((doc.metadata.get("benchmark_resource_id") or "").encode("utf-8"))
        digest.update((doc.metadata.get("file_id") or "").encode("utf-8"))
        digest.update(doc.page_content.encode("utf-8"))
    return digest.hexdigest()


def _cache_paths(cache_dir: Path, spec: ModelSpec) -> tuple[Path, Path]:
    model_dir = cache_dir / spec.key
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir / "corpus_embeddings.npy", model_dir / "corpus_meta.json"


def _load_cached_embeddings(
    cache_dir: Path,
    spec: ModelSpec,
    docs_sig: str,
) -> Optional[np.ndarray]:
    emb_path, meta_path = _cache_paths(cache_dir, spec)
    if not emb_path.exists() or not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("docs_signature") != docs_sig:
            return None
        if meta.get("model_name") != spec.hf_id:
            return None
        if meta.get("doc_prefix") != spec.doc_prefix:
            return None
        return np.load(emb_path)
    except Exception:
        return None


def _save_cached_embeddings(
    cache_dir: Path,
    spec: ModelSpec,
    docs_sig: str,
    embeddings: np.ndarray,
) -> None:
    emb_path, meta_path = _cache_paths(cache_dir, spec)
    np.save(emb_path, embeddings)
    meta_path.write_text(
        json.dumps(
            {
                "model_name": spec.hf_id,
                "doc_prefix": spec.doc_prefix,
                "docs_signature": docs_sig,
                "shape": list(embeddings.shape),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _doc_matches_resource(doc: Document, expected_resource_ids: List[str]) -> bool:
    resource_id = doc.metadata.get("benchmark_resource_id")
    return resource_id in expected_resource_ids


def _doc_matches_evidence(doc: Document, evidence_groups_norm: List[List[str]]) -> bool:
    text = normalize_for_match(doc.page_content)
    return any(any(term in text for term in group) for group in evidence_groups_norm)


def _find_first_rank(results: List[Document], predicate) -> Optional[int]:
    for idx, doc in enumerate(results, start=1):
        if predicate(doc):
            return idx
    return None


def _coverage(results: List[Document], evidence_groups_norm: List[List[str]]) -> float:
    if not evidence_groups_norm:
        return 1.0
    combined_text = "\n".join(normalize_for_match(doc.page_content) for doc in results)
    covered = 0
    for group in evidence_groups_norm:
        if any(term in combined_text for term in group):
            covered += 1
    return covered / len(evidence_groups_norm)


def _preview(text: str, length: int = 180) -> str:
    compact = " ".join(text.strip().split())
    return compact[:length]


def _safe_model_size_bytes(model_id: str) -> Optional[int]:
    try:
        info = model_info(model_id)
    except Exception:
        return None
    total = 0
    found = False
    for sibling in info.siblings:
        if sibling.rfilename.endswith((".safetensors", ".bin", ".onnx")) and sibling.size:
            total += int(sibling.size)
            found = True
    return total if found else None


def benchmark_model(
    spec: ModelSpec,
    docs: List[Document],
    cases: List[Dict[str, Any]],
    cache_dir: Path,
    top_k: int,
    batch_size: int,
) -> Dict[str, Any]:
    docs_sig = corpus_signature(docs)
    sampler = MemorySampler()
    sampler.start()
    started = time.perf_counter()
    load_seconds = 0.0
    embed_seconds = 0.0
    query_seconds = 0.0

    try:
        model_load_start = time.perf_counter()
        embedder = BenchmarkEmbedder(spec)
        load_seconds = time.perf_counter() - model_load_start

        embeddings = _load_cached_embeddings(cache_dir, spec, docs_sig)
        if embeddings is None:
            doc_embed_start = time.perf_counter()
            embeddings = embedder.embed_documents_np([doc.page_content for doc in docs], batch_size=batch_size)
            embed_seconds = time.perf_counter() - doc_embed_start
            _save_cached_embeddings(cache_dir, spec, docs_sig, embeddings)

        store = FAISSVectorStore([], embedder)  # type: ignore[arg-type]
        store.add_documents_with_embeddings(docs, embeddings, rebuild_index=True)

        case_results: List[Dict[str, Any]] = []
        source_mrr_total = 0.0
        evidence_mrr_total = 0.0
        coverage_total = 0.0
        source_hits = {1: 0, 3: 0, 5: 0}
        evidence_hits = {1: 0, 3: 0, 5: 0}

        for case in cases:
            limit = int(case.get("top_k", top_k))
            query_start = time.perf_counter()
            results = store.retrieve(case["query"], top_k=limit)
            query_seconds += time.perf_counter() - query_start

            evidence_groups_norm = [
                [normalize_for_match(term) for term in group]
                for group in case.get("evidence_groups", [])
            ]
            expected_resources = list(case.get("expected_resource_ids", []))

            source_rank = _find_first_rank(
                results,
                lambda doc: _doc_matches_resource(doc, expected_resources),
            )
            evidence_rank = _find_first_rank(
                results,
                lambda doc: _doc_matches_resource(doc, expected_resources) and _doc_matches_evidence(doc, evidence_groups_norm),
            )
            coverage = _coverage(results, evidence_groups_norm)

            if source_rank:
                source_mrr_total += 1.0 / source_rank
                for cutoff in source_hits:
                    if source_rank <= cutoff:
                        source_hits[cutoff] += 1
            if evidence_rank:
                evidence_mrr_total += 1.0 / evidence_rank
                for cutoff in evidence_hits:
                    if evidence_rank <= cutoff:
                        evidence_hits[cutoff] += 1

            coverage_total += coverage

            case_results.append(
                {
                    "id": case["id"],
                    "category": case["category"],
                    "query": case["query"],
                    "source_rank": source_rank,
                    "evidence_rank": evidence_rank,
                    "coverage_top_k": round(coverage, 4),
                    "top_results": [
                        {
                            "rank": idx + 1,
                            "resource_id": doc.metadata.get("benchmark_resource_id"),
                            "file_id": doc.metadata.get("file_id"),
                            "page": doc.metadata.get("page"),
                            "preview": _preview(doc.page_content),
                        }
                        for idx, doc in enumerate(results[:limit])
                    ],
                }
            )

        total_cases = len(cases)
        peak_rss_gb = sampler.stop() / (1024**3)
        total_elapsed = time.perf_counter() - started
        model_bytes = _safe_model_size_bytes(spec.hf_id)
        summary = {
            "model_key": spec.key,
            "model_name": spec.hf_id,
            "embedding_dim": getattr(embedder, "embedding_dim", None),
            "model_artifact_gb": round(model_bytes / (1024**3), 3) if model_bytes else None,
            "peak_process_rss_gb": round(peak_rss_gb, 3),
            "load_seconds": round(load_seconds, 3),
            "doc_embed_seconds": round(embed_seconds, 3),
            "query_seconds_total": round(query_seconds, 3),
            "elapsed_seconds": round(total_elapsed, 3),
            "source_mrr": round(source_mrr_total / total_cases, 4),
            "evidence_mrr": round(evidence_mrr_total / total_cases, 4),
            "coverage_top_k": round(coverage_total / total_cases, 4),
            "source_hit_rate_at_1": round(source_hits[1] / total_cases, 4),
            "source_hit_rate_at_3": round(source_hits[3] / total_cases, 4),
            "source_hit_rate_at_5": round(source_hits[5] / total_cases, 4),
            "evidence_hit_rate_at_1": round(evidence_hits[1] / total_cases, 4),
            "evidence_hit_rate_at_3": round(evidence_hits[3] / total_cases, 4),
            "evidence_hit_rate_at_5": round(evidence_hits[5] / total_cases, 4),
        }
        summary["composite_score"] = round(
            (0.4 * summary["coverage_top_k"])
            + (0.3 * summary["source_mrr"])
            + (0.3 * summary["evidence_mrr"]),
            4,
        )
        return {
            "summary": summary,
            "cases": case_results,
        }
    finally:
        try:
            sampler.stop()
        except Exception:
            pass
        gc.collect()


def write_markdown_report(
    report_path: Path,
    config: Dict[str, Any],
    model_reports: List[Dict[str, Any]],
) -> None:
    lines: List[str] = []
    generated_at = datetime.now().isoformat(timespec="seconds")
    lines.append(f"# Embedding Benchmark Report\n")
    lines.append(f"- Generated at: `{generated_at}`")
    lines.append(f"- Corpus resources: {len(config['resources'])}")
    lines.append(f"- Query cases: {len(config['cases'])}\n")

    lines.append("## Summary\n")
    lines.append("| Model | Composite | Coverage@k | Source MRR | Evidence MRR | Hit@1 src | Hit@1 ev | Peak RSS (GB) | Artifact (GB) |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for report in sorted(model_reports, key=lambda item: item["summary"]["composite_score"], reverse=True):
        s = report["summary"]
        lines.append(
            f"| `{s['model_key']}` | {s['composite_score']:.4f} | {s['coverage_top_k']:.4f} | "
            f"{s['source_mrr']:.4f} | {s['evidence_mrr']:.4f} | {s['source_hit_rate_at_1']:.4f} | "
            f"{s['evidence_hit_rate_at_1']:.4f} | {s['peak_process_rss_gb']:.3f} | "
            f"{s['model_artifact_gb'] if s['model_artifact_gb'] is not None else 'n/a'} |"
        )

    lines.append("\n## Hard Cases\n")
    for report in model_reports:
        s = report["summary"]
        lines.append(f"### `{s['model_key']}`\n")
        hard_cases = [
            case
            for case in report["cases"]
            if (case["coverage_top_k"] < 1.0) or (case["source_rank"] not in (1, 2, 3))
        ]
        if not hard_cases:
            lines.append("- No notable misses.\n")
            continue
        for case in hard_cases[:8]:
            lines.append(
                f"- `{case['id']}`: source_rank={case['source_rank']} evidence_rank={case['evidence_rank']} "
                f"coverage={case['coverage_top_k']}"
            )
            for result in case["top_results"][:3]:
                lines.append(
                    f"  - #{result['rank']} `{result['resource_id']}` p.{result['page']}: {result['preview']}"
                )
        lines.append("")

    report_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark embedding models on the current LLM Learning retrieval corpus.")
    parser.add_argument(
        "--cases",
        default=str(REPO_ROOT / "benchmarks" / "embedding_model_cases.json"),
        help="Path to benchmark cases JSON.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "aiteamvn",
            "gte-multilingual-base",
            "multilingual-e5-small",
            "bkai-vietnamese-bi-encoder",
            "dangvantuan-vietnamese-embedding",
        ],
        help="Model keys to benchmark.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Default top-k retrieval cutoff.")
    parser.add_argument("--batch-size", type=int, default=16, help="Embedding batch size.")
    parser.add_argument(
        "--cache-dir",
        default=str(REPO_ROOT / "tmp" / "embedding_benchmark_cache"),
        help="Directory for benchmark embedding cache.",
    )
    parser.add_argument(
        "--report-json",
        default=str(REPO_ROOT / "reports" / f"embedding_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"),
        help="Path to write detailed JSON report.",
    )
    parser.add_argument(
        "--report-md",
        default=str(REPO_ROOT / "reports" / f"embedding_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"),
        help="Path to write Markdown summary report.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases_path = Path(args.cases)
    config = _read_cases(cases_path)
    docs, _resources = load_corpus(REPO_ROOT, config)

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    requested_models: List[ModelSpec] = []
    for key in args.models:
        if key not in MODEL_SPECS:
            raise KeyError(f"Unknown model key: {key}")
        requested_models.append(MODEL_SPECS[key])

    model_reports: List[Dict[str, Any]] = []
    for spec in requested_models:
        print(f"[benchmark] model={spec.key} hf_id={spec.hf_id}")
        report = benchmark_model(
            spec=spec,
            docs=docs,
            cases=config["cases"],
            cache_dir=cache_dir,
            top_k=int(args.top_k),
            batch_size=int(args.batch_size),
        )
        model_reports.append(report)
        summary = report["summary"]
        print(
            "[benchmark] done model=%s composite=%.4f coverage=%.4f source_mrr=%.4f evidence_mrr=%.4f peak_rss=%.3fGB"
            % (
                summary["model_key"],
                summary["composite_score"],
                summary["coverage_top_k"],
                summary["source_mrr"],
                summary["evidence_mrr"],
                summary["peak_process_rss_gb"],
            )
        )

    report_json = Path(args.report_json)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "cases_path": str(cases_path),
        "resources": config["resources"],
        "cases": config["cases"],
        "models": model_reports,
    }
    report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    report_md = Path(args.report_md)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    write_markdown_report(report_md, config, model_reports)

    print(f"[benchmark] json_report={report_json}")
    print(f"[benchmark] md_report={report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
