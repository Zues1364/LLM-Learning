from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class WeightCandidate:
    key_weight: float
    row_weight: float
    cell_f1_weight: float
    winner: str
    runner_up: str
    winner_score: float
    runner_up_score: float
    gap_to_second: float
    ranking: List[Tuple[str, float]]
    distance_to_balanced: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep PDF composite-score weights over an existing PDF benchmark report."
    )
    parser.add_argument(
        "--input-json",
        default=str(REPO_ROOT / "reports" / "pdf_extraction_benchmark_latest.json"),
        help="Detailed PDF extraction benchmark JSON report.",
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
        "--top-n",
        type=int,
        default=25,
        help="Number of top gap-ranked candidates to include in reports. Default: 25",
    )
    parser.add_argument(
        "--report-json",
        default=str(REPO_ROOT / "reports" / "pdf_score_weight_sweep_latest.json"),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--report-md",
        default=str(REPO_ROOT / "reports" / "pdf_score_weight_sweep_latest.md"),
        help="Output Markdown path.",
    )
    return parser.parse_args()


def _round_weight(value: float) -> float:
    return round(value + 1e-12, 4)


def _load_methods(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("summary_by_method")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Không đọc được danh sách phương pháp từ {path}")
    return rows


def _extract_metrics(method_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    metrics: List[Dict[str, Any]] = []
    for row in method_rows:
        metrics.append(
            {
                "method": str(row["method"]),
                "key_field_accuracy": float(row["key_field_accuracy"]),
                "row_exact_accuracy": float(row["row_exact_accuracy"]),
                "cell_f1": float(row["cell_f1"]),
            }
        )
    return metrics


def _sweep_candidates(
    method_metrics: List[Dict[str, Any]],
    step: float,
    min_weight: float,
) -> List[WeightCandidate]:
    candidates: List[WeightCandidate] = []
    step_count = int(round(1.0 / step))
    for key_idx in range(step_count + 1):
        key_weight = _round_weight(key_idx * step)
        for row_idx in range(step_count + 1 - key_idx):
            row_weight = _round_weight(row_idx * step)
            cell_f1_weight = _round_weight(1.0 - key_weight - row_weight)
            if cell_f1_weight < 0:
                continue
            if min(key_weight, row_weight, cell_f1_weight) < min_weight:
                continue

            scores: List[Tuple[str, float]] = []
            for metric in method_metrics:
                score = (
                    key_weight * metric["key_field_accuracy"]
                    + row_weight * metric["row_exact_accuracy"]
                    + cell_f1_weight * metric["cell_f1"]
                )
                scores.append((metric["method"], round(score, 6)))

            ranking = sorted(scores, key=lambda item: item[1], reverse=True)
            winner, winner_score = ranking[0]
            runner_up, runner_up_score = ranking[1]
            gap_to_second = round(winner_score - runner_up_score, 6)
            distance_to_balanced = round(
                (key_weight - (1 / 3)) ** 2
                + (row_weight - (1 / 3)) ** 2
                + (cell_f1_weight - (1 / 3)) ** 2,
                6,
            )
            candidates.append(
                WeightCandidate(
                    key_weight=key_weight,
                    row_weight=row_weight,
                    cell_f1_weight=cell_f1_weight,
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
            -candidate.key_weight,
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
        "key_weight": candidate.key_weight,
        "row_weight": candidate.row_weight,
        "cell_f1_weight": candidate.cell_f1_weight,
        "winner": candidate.winner,
        "runner_up": candidate.runner_up,
        "winner_score": candidate.winner_score,
        "runner_up_score": candidate.runner_up_score,
        "gap_to_second": candidate.gap_to_second,
        "distance_to_balanced": candidate.distance_to_balanced,
        "ranking": [
            {"method": method, "score": score}
            for method, score in candidate.ranking
        ],
    }


def _candidate_identity(candidate: WeightCandidate) -> Tuple[float, float, float]:
    return (
        candidate.key_weight,
        candidate.row_weight,
        candidate.cell_f1_weight,
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
    method_metrics: List[Dict[str, Any]],
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
    lines.append("# Báo cáo quét trọng số cho điểm tổng hợp trích xuất PDF")
    lines.append("")
    lines.append(f"- Nguồn số liệu: `{input_json}`")
    lines.append(f"- Số phương pháp đem so: {len(method_metrics)}")
    lines.append(f"- Bước quét trọng số: {step:.2f}")
    lines.append(f"- Trọng số tối thiểu mỗi chỉ số: {min_weight:.2f}")
    lines.append(
        f"- Vùng ổn định: các cấu hình có khoảng cách giữa phương pháp đứng đầu và phương pháp thứ hai đạt ít nhất {stable_gap_ratio:.0%} so với cấu hình tách tốt nhất."
    )
    lines.append("")
    lines.append("## Bộ trọng số đề xuất")
    lines.append("")
    lines.append(
        f"- Đề xuất: `Cột chính = {recommended.key_weight:.2f}`, `Hàng = {recommended.row_weight:.2f}`, `F1 ô = {recommended.cell_f1_weight:.2f}`"
    )
    lines.append(f"- Phương pháp đứng đầu dưới bộ trọng số này: `{recommended.winner}`")
    lines.append(f"- Khoảng cách với phương pháp đứng thứ hai: `{recommended.gap_to_second:.4f}`")
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
            f"- Hiện tại: `0.40 / 0.35 / 0.25`, phương pháp đứng đầu `{current_candidate.winner}`, khoảng cách `{current_candidate.gap_to_second:.4f}`"
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
    lines.append("| # | Cột chính | Hàng | F1 ô | Phương pháp đứng đầu | Cách biệt |")
    lines.append("| ---: | ---: | ---: | ---: | --- | ---: |")
    for idx, candidate in enumerate(top_candidates, start=1):
        lines.append(
            f"| {idx} | {candidate.key_weight:.2f} | {candidate.row_weight:.2f} | {candidate.cell_f1_weight:.2f} | "
            f"`{candidate.winner}` | {candidate.gap_to_second:.4f} |"
        )
    lines.append("")
    lines.append("## Tần suất phương pháp đứng đầu trong toàn bộ miền quét")
    lines.append("")
    for method, count in sorted(metadata["winner_frequency"].items(), key=lambda item: item[1], reverse=True):
        lines.append(f"- `{method}`: {count} cấu hình")
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_json = Path(args.input_json)
    report_json = Path(args.report_json)
    report_md = Path(args.report_md)

    method_rows = _load_methods(input_json)
    method_metrics = _extract_metrics(method_rows)
    candidates = _sweep_candidates(
        method_metrics=method_metrics,
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
            if candidate.key_weight == 0.40
            and candidate.row_weight == 0.35
            and candidate.cell_f1_weight == 0.25
        ),
        None,
    )

    payload = {
        "input_json": str(input_json),
        "constraints": {
            "step": float(args.step),
            "min_weight": float(args.min_weight),
            "stable_gap_ratio": float(args.stable_gap_ratio),
            "top_n": int(args.top_n),
        },
        "recommended": _candidate_to_dict(recommended),
        "current_default": _candidate_to_dict(current_candidate) if current_candidate else None,
        "metadata": metadata,
        "recommended_gap_rank": recommended_gap_rank,
        "method_metrics": method_metrics,
        "top_candidates": [_candidate_to_dict(candidate) for candidate in top_candidates],
        "all_candidates": [_candidate_to_dict(candidate) for candidate in candidates],
    }
    report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    _write_markdown(
        path=report_md,
        input_json=input_json,
        method_metrics=method_metrics,
        recommended=recommended,
        metadata=metadata,
        top_candidates=top_candidates,
        current_candidate=current_candidate,
        recommended_gap_rank=recommended_gap_rank,
        min_weight=float(args.min_weight),
        step=float(args.step),
        stable_gap_ratio=float(args.stable_gap_ratio),
    )

    print(
        "[pdf-weights] recommended="
        f"{recommended.key_weight:.2f}/{recommended.row_weight:.2f}/{recommended.cell_f1_weight:.2f}"
    )
    print(f"[pdf-weights] json_report={report_json}")
    print(f"[pdf-weights] md_report={report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
