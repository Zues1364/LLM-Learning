from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class WeightCandidate:
    coverage_weight: float
    source_weight: float
    evidence_weight: float
    winner: str
    runner_up: str
    winner_score: float
    runner_up_score: float
    gap_to_second: float
    ranking: List[Tuple[str, float]]
    distance_to_balanced: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep retrieval composite-score weights over an existing embedding benchmark report."
    )
    parser.add_argument(
        "--input-json",
        default=str(REPO_ROOT / "reports" / "embedding_benchmark_compare_20260515.json"),
        help="Detailed embedding benchmark JSON report.",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.05,
        help="Weight step size. Default: 0.05",
    )
    parser.add_argument(
        "--min-weight",
        type=float,
        default=0.20,
        help="Minimum allowed weight for each metric. Default: 0.20",
    )
    parser.add_argument(
        "--stable-gap-ratio",
        type=float,
        default=0.85,
        help="Minimum gap ratio versus the best candidate to be considered in the stable region. Default: 0.85",
    )
    parser.add_argument(
        "--report-json",
        default=str(REPO_ROOT / "reports" / "retrieval_weight_sweep_latest.json"),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--report-md",
        default=str(REPO_ROOT / "reports" / "retrieval_weight_sweep_latest.md"),
        help="Output Markdown path.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=25,
        help="Number of top gap-ranked candidates to include in reports. Default: 25",
    )
    return parser.parse_args()


def _round_weight(value: float) -> float:
    return round(value + 1e-12, 4)


def _load_models(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    models = payload.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError(f"Không đọc được danh sách mô hình từ {path}")
    return models


def _extract_metrics(model_reports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    metrics: List[Dict[str, Any]] = []
    for model in model_reports:
        summary = model["summary"]
        metrics.append(
            {
                "model_key": str(summary["model_key"]),
                "coverage_top_k": float(summary["coverage_top_k"]),
                "source_mrr": float(summary["source_mrr"]),
                "evidence_mrr": float(summary["evidence_mrr"]),
            }
        )
    return metrics


def _sweep_candidates(
    model_metrics: List[Dict[str, Any]],
    step: float,
    min_weight: float,
) -> List[WeightCandidate]:
    candidates: List[WeightCandidate] = []
    step_count = int(round(1.0 / step))
    for cov_idx in range(step_count + 1):
        coverage_weight = _round_weight(cov_idx * step)
        for src_idx in range(step_count + 1 - cov_idx):
            source_weight = _round_weight(src_idx * step)
            evidence_weight = _round_weight(1.0 - coverage_weight - source_weight)
            if evidence_weight < 0:
                continue
            if min(coverage_weight, source_weight, evidence_weight) < min_weight:
                continue
            if coverage_weight < source_weight or coverage_weight < evidence_weight:
                continue

            scores: List[Tuple[str, float]] = []
            for metric in model_metrics:
                score = (
                    coverage_weight * metric["coverage_top_k"]
                    + source_weight * metric["source_mrr"]
                    + evidence_weight * metric["evidence_mrr"]
                )
                scores.append((metric["model_key"], round(score, 6)))

            ranking = sorted(scores, key=lambda item: item[1], reverse=True)
            winner, winner_score = ranking[0]
            runner_up, runner_up_score = ranking[1]
            gap_to_second = round(winner_score - runner_up_score, 6)
            distance_to_balanced = round(
                (coverage_weight - (1 / 3)) ** 2
                + (source_weight - (1 / 3)) ** 2
                + (evidence_weight - (1 / 3)) ** 2,
                6,
            )
            candidates.append(
                WeightCandidate(
                    coverage_weight=coverage_weight,
                    source_weight=source_weight,
                    evidence_weight=evidence_weight,
                    winner=winner,
                    runner_up=runner_up,
                    winner_score=winner_score,
                    runner_up_score=runner_up_score,
                    gap_to_second=gap_to_second,
                    ranking=ranking,
                    distance_to_balanced=distance_to_balanced,
                )
            )
    return candidates


def _select_recommended(
    candidates: List[WeightCandidate],
    stable_gap_ratio: float,
) -> Tuple[WeightCandidate, Dict[str, Any]]:
    if not candidates:
        raise ValueError("Không có cấu hình trọng số nào thỏa điều kiện quét.")

    best_gap = max(candidate.gap_to_second for candidate in candidates)
    stable_gap_threshold = round(best_gap * stable_gap_ratio, 6)
    stable_region = [
        candidate
        for candidate in candidates
        if candidate.gap_to_second >= stable_gap_threshold
    ]

    recommended = sorted(
        stable_region,
        key=lambda candidate: (
            candidate.distance_to_balanced,
            -candidate.gap_to_second,
            -candidate.coverage_weight,
        ),
    )[0]

    winner_frequency: Dict[str, int] = {}
    for candidate in candidates:
        winner_frequency[candidate.winner] = winner_frequency.get(candidate.winner, 0) + 1

    metadata = {
        "best_gap": round(best_gap, 6),
        "stable_gap_threshold": stable_gap_threshold,
        "stable_region_count": len(stable_region),
        "winner_frequency": winner_frequency,
    }
    return recommended, metadata


def _candidate_to_dict(candidate: WeightCandidate) -> Dict[str, Any]:
    return {
        "coverage_weight": candidate.coverage_weight,
        "source_weight": candidate.source_weight,
        "evidence_weight": candidate.evidence_weight,
        "winner": candidate.winner,
        "runner_up": candidate.runner_up,
        "winner_score": candidate.winner_score,
        "runner_up_score": candidate.runner_up_score,
        "gap_to_second": candidate.gap_to_second,
        "distance_to_balanced": candidate.distance_to_balanced,
        "ranking": [
            {"model_key": model_key, "score": score}
            for model_key, score in candidate.ranking
        ],
    }


def _candidate_identity(candidate: WeightCandidate) -> Tuple[float, float, float]:
    return (
        candidate.coverage_weight,
        candidate.source_weight,
        candidate.evidence_weight,
    )


def _gap_rank(candidates: List[WeightCandidate], target: WeightCandidate) -> int:
    ordered = sorted(candidates, key=lambda candidate: candidate.gap_to_second, reverse=True)
    target_id = _candidate_identity(target)
    for index, candidate in enumerate(ordered, start=1):
        if _candidate_identity(candidate) == target_id:
            return index
    raise ValueError("Không tìm thấy cấu hình cần xếp hạng theo khoảng cách.")


def _write_markdown(
    path: Path,
    input_json: Path,
    model_metrics: List[Dict[str, Any]],
    recommended: WeightCandidate,
    metadata: Dict[str, Any],
    top_candidates: List[WeightCandidate],
    current_candidate: WeightCandidate | None,
    recommended_gap_rank: int,
    min_weight: float,
    step: float,
    stable_gap_ratio: float,
) -> None:
    lines: List[str] = []
    lines.append("# Báo cáo quét trọng số cho điểm tổng hợp truy hồi")
    lines.append("")
    lines.append(f"- Nguồn số liệu: `{input_json}`")
    lines.append(f"- Số mô hình đem so: {len(model_metrics)}")
    lines.append(f"- Bước quét trọng số: {step:.2f}")
    lines.append(f"- Trọng số tối thiểu mỗi chỉ số: {min_weight:.2f}")
    lines.append("- Ràng buộc miền quét: trọng số độ phủ không nhỏ hơn hai trọng số MRR.")
    lines.append(
        f"- Vùng ổn định: các cấu hình có khoảng cách giữa mô hình đứng đầu và mô hình thứ hai đạt ít nhất {stable_gap_ratio:.0%} so với cấu hình tách tốt nhất."
    )
    lines.append("")
    lines.append("## Bộ trọng số đề xuất")
    lines.append("")
    lines.append(
        f"- Đề xuất: `Coverage@5 = {recommended.coverage_weight:.2f}`, `MRR tài liệu = {recommended.source_weight:.2f}`, `MRR đoạn = {recommended.evidence_weight:.2f}`"
    )
    lines.append(f"- Mô hình đứng đầu dưới bộ trọng số này: `{recommended.winner}`")
    lines.append(f"- Khoảng cách với mô hình đứng thứ hai: `{recommended.gap_to_second:.4f}`")
    lines.append(
        f"- Khoảng cách của cấu hình này tới phân bố cân bằng `(1/3, 1/3, 1/3)`: `{recommended.distance_to_balanced:.6f}`"
    )
    lines.append(
        f"- Xếp hạng theo khoảng cách nếu chỉ nhìn `gap_to_second`: `{recommended_gap_rank}/{sum(metadata['winner_frequency'].values())}`"
    )
    lines.append(
        "- Bộ này được chọn vì là cấu hình cân bằng nhất trong vùng ổn định, không phải vì có khoảng cách lớn nhất."
    )
    if current_candidate:
        lines.append("")
        lines.append("## So với bộ trọng số đang dùng")
        lines.append("")
        lines.append(
            f"- Hiện tại: `0.40 / 0.30 / 0.30`, mô hình đứng đầu `{current_candidate.winner}`, khoảng cách `{current_candidate.gap_to_second:.4f}`"
        )
        lines.append(
            f"- Chênh lệch khoảng cách so với bộ đề xuất: `{recommended.gap_to_second - current_candidate.gap_to_second:+.4f}`"
        )
    lines.append("")
    lines.append("## Các cấu hình đứng đầu theo khoảng cách")
    lines.append("")
    lines.append(
        "Bảng dưới được sắp theo `gap_to_second` giảm dần. Vì vậy, cấu hình được đề xuất có thể không xuất hiện trong nhóm đầu nếu nó được chọn theo tiêu chí cân bằng trong vùng ổn định."
    )
    lines.append("")
    lines.append("| # | Coverage@5 | MRR tài liệu | MRR đoạn | Mô hình đứng đầu | Cách biệt |")
    lines.append("| ---: | ---: | ---: | ---: | --- | ---: |")
    for idx, candidate in enumerate(top_candidates, start=1):
        lines.append(
            f"| {idx} | {candidate.coverage_weight:.2f} | {candidate.source_weight:.2f} | {candidate.evidence_weight:.2f} | "
            f"`{candidate.winner}` | {candidate.gap_to_second:.4f} |"
        )
    lines.append("")
    lines.append("## Tần suất mô hình đứng đầu trong toàn bộ miền quét")
    lines.append("")
    for model_key, count in sorted(metadata["winner_frequency"].items(), key=lambda item: item[1], reverse=True):
        lines.append(f"- `{model_key}`: {count} cấu hình")

    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_json = Path(args.input_json)
    report_json = Path(args.report_json)
    report_md = Path(args.report_md)

    models = _load_models(input_json)
    model_metrics = _extract_metrics(models)
    candidates = _sweep_candidates(
        model_metrics=model_metrics,
        step=float(args.step),
        min_weight=float(args.min_weight),
    )
    recommended, metadata = _select_recommended(
        candidates=candidates,
        stable_gap_ratio=float(args.stable_gap_ratio),
    )
    top_candidates = sorted(
        candidates,
        key=lambda candidate: candidate.gap_to_second,
        reverse=True,
    )[: int(args.top_n)]
    recommended_gap_rank = _gap_rank(candidates, recommended)
    current_candidate = next(
        (
            candidate
            for candidate in candidates
            if candidate.coverage_weight == 0.40
            and candidate.source_weight == 0.30
            and candidate.evidence_weight == 0.30
        ),
        None,
    )

    payload = {
        "input_json": str(input_json),
        "constraints": {
            "step": float(args.step),
            "min_weight": float(args.min_weight),
            "coverage_not_less_than_mrr": True,
            "stable_gap_ratio": float(args.stable_gap_ratio),
            "top_n": int(args.top_n),
        },
        "recommended": _candidate_to_dict(recommended),
        "current_default": _candidate_to_dict(current_candidate) if current_candidate else None,
        "metadata": metadata,
        "recommended_gap_rank": recommended_gap_rank,
        "model_metrics": model_metrics,
        "top_candidates": [_candidate_to_dict(candidate) for candidate in top_candidates],
        "all_candidates": [_candidate_to_dict(candidate) for candidate in candidates],
    }
    report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    _write_markdown(
        path=report_md,
        input_json=input_json,
        model_metrics=model_metrics,
        recommended=recommended,
        metadata=metadata,
        top_candidates=top_candidates,
        current_candidate=current_candidate,
        recommended_gap_rank=recommended_gap_rank,
        min_weight=float(args.min_weight),
        step=float(args.step),
        stable_gap_ratio=float(args.stable_gap_ratio),
    )

    print(f"[weights] recommended={recommended.coverage_weight:.2f}/{recommended.source_weight:.2f}/{recommended.evidence_weight:.2f}")
    print(f"[weights] json_report={report_json}")
    print(f"[weights] md_report={report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
