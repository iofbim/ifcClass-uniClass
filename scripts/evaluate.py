import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple

from etl.etl_map import connect, load_settings


def _latest_run_id(conn) -> Optional[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT run_id::text, max(generated_at) AS last_ts
            FROM ifc_uniclass_candidate_history
            GROUP BY run_id
            ORDER BY last_ts DESC
            LIMIT 1
            """
        )
        row = cur.fetchone()
    return row["run_id"] if row else None


def _fetch_candidates(conn, run_id: str) -> Tuple[Dict[Tuple[str, str], dict], Optional[str]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT ifc_id, code, facet, score, lexical_score, embedding_score,
                   feature_multiplier, discipline_multiplier, token_overlap_multiplier,
                   anchor_applied, source_uniclass_revision
            FROM ifc_uniclass_candidate_history
            WHERE run_id = %s
            """,
            (run_id,),
        )
        rows = cur.fetchall()
    candidates: Dict[Tuple[str, str], dict] = {}
    revision: Optional[str] = None
    for row in rows:
        key = (str(row["ifc_id"]), str(row["code"]))
        if revision is None:
            revision = row.get("source_uniclass_revision")
        prev = candidates.get(key)
        if prev is None or (row.get("score") or 0) > (prev.get("score") or 0):
            candidates[key] = row
    return candidates, revision


def _fetch_decisions(conn, revision: Optional[str]):
    with conn.cursor() as cur:
        if revision:
            cur.execute(
                """
                SELECT ifc_id, code, relation_type, decision, facet, source_uniclass_revision
                FROM review_decision
                WHERE source_uniclass_revision = %s OR source_uniclass_revision IS NULL
                """,
                (revision,),
            )
        else:
            cur.execute(
                """
                SELECT ifc_id, code, relation_type, decision, facet, source_uniclass_revision
                FROM review_decision
                """
            )
        return cur.fetchall()


def _init_counts() -> Dict[str, int]:
    return {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "reviewed": 0}


def _compute_metrics(counts: Dict[str, int]) -> Dict[str, Optional[float]]:
    tp = counts.get("tp", 0)
    fp = counts.get("fp", 0)
    fn = counts.get("fn", 0)
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def _sum_dict(d: Dict[str, int]) -> int:
    return sum(d.values())


def evaluate_run(conn, run_id: str, threshold: float):
    candidates, revision = _fetch_candidates(conn, run_id)
    if not candidates:
        raise RuntimeError(f"No candidate history found for run_id={run_id}")

    decisions = _fetch_decisions(conn, revision)
    counts_by_facet = defaultdict(_init_counts)
    overall_counts = _init_counts()
    unreviewed_pred = defaultdict(int)
    missing_candidates = defaultdict(int)
    reviewed_pairs = set()

    for dec in decisions:
        ifc_id = str(dec["ifc_id"])
        code = str(dec["code"])
        decision = (dec.get("decision") or "").strip().lower()
        if decision not in {"accept", "reject"}:
            continue
        key = (ifc_id, code)
        cand = candidates.get(key)
        facet = dec.get("facet") or (cand.get("facet") if cand else "") or "UNKNOWN"
        counts = counts_by_facet[facet]
        counts["reviewed"] += 1
        overall_counts["reviewed"] += 1
        score = cand.get("score") if cand else None
        pred_positive = score is not None and score >= threshold
        missing = cand is None
        if decision == "accept":
            if pred_positive:
                counts["tp"] += 1
                overall_counts["tp"] += 1
            else:
                counts["fn"] += 1
                overall_counts["fn"] += 1
        else:  # reject
            if pred_positive:
                counts["fp"] += 1
                overall_counts["fp"] += 1
            else:
                counts["tn"] += 1
                overall_counts["tn"] += 1
        if missing:
            missing_candidates[facet] += 1
        reviewed_pairs.add(key)

    for key, cand in candidates.items():
        score = cand.get("score") or 0.0
        if score >= threshold and key not in reviewed_pairs:
            facet = cand.get("facet") or "UNKNOWN"
            unreviewed_pred[facet] += 1

    metrics_by_facet = {}
    for facet, counts in counts_by_facet.items():
        metrics = _compute_metrics(counts)
        metrics_by_facet[facet] = {
            **counts,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
        }
    overall_metrics = _compute_metrics(overall_counts)

    return {
        "run_id": run_id,
        "revision": revision,
        "counts_by_facet": {facet: dict(v) for facet, v in metrics_by_facet.items()},
        "overall_counts": dict(overall_counts),
        "overall_metrics": overall_metrics,
        "unreviewed_predictions": {"by_facet": dict(unreviewed_pred), "total": _sum_dict(unreviewed_pred)},
        "decisions_missing_candidates": {"by_facet": dict(missing_candidates), "total": _sum_dict(missing_candidates)},
        "reviewed_pairs": reviewed_pairs,
    }


def _format_metric(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def build_report(current, baseline, threshold: float):
    report = {
        "run_id": current["run_id"],
        "revision": current["revision"],
        "threshold": threshold,
        "overall": {
            **{k: current["overall_counts"].get(k, 0) for k in ("tp", "fp", "tn", "fn", "reviewed")},
            **current["overall_metrics"],
        },
        "facets": current["counts_by_facet"],
        "unreviewed_predictions": current["unreviewed_predictions"],
        "decisions_missing_candidates": current["decisions_missing_candidates"],
    }
    if baseline:
        report["baseline_comparison"] = {
            "baseline_run_id": baseline["run_id"],
            "precision_delta": None,
            "recall_delta": None,
        }
        cur_prec = current["overall_metrics"].get("precision")
        base_prec = baseline["overall_metrics"].get("precision")
        if cur_prec is not None and base_prec is not None:
            report["baseline_comparison"]["precision_delta"] = cur_prec - base_prec
        cur_rec = current["overall_metrics"].get("recall")
        base_rec = baseline["overall_metrics"].get("recall")
        if cur_rec is not None and base_rec is not None:
            report["baseline_comparison"]["recall_delta"] = cur_rec - base_rec
    return report


def write_outputs(report, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    md_path = out_path.with_suffix(".md")
    lines = [
        f"# Evaluation Report for {report['run_id']}",
        "",
        f"- Uniclass revision: {report.get('revision') or 'unknown'}",
        f"- Acceptance threshold: {report['threshold']:.2f}",
        "",
        "## Overall",
        f"- Precision: {_format_metric(report['overall'].get('precision'))}",
        f"- Recall: {_format_metric(report['overall'].get('recall'))}",
        f"- F1: {_format_metric(report['overall'].get('f1'))}",
        f"- Reviewed pairs: {report['overall'].get('reviewed', 0)}",
        f"- TP/FP/FN/TN: {report['overall'].get('tp', 0)}/{report['overall'].get('fp', 0)}/{report['overall'].get('fn', 0)}/{report['overall'].get('tn', 0)}",
        "",
        "## Facet Metrics",
        "| Facet | Precision | Recall | F1 | Reviewed | TP | FP | FN |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for facet, stats in sorted(report.get("facets", {}).items()):
        lines.append(
            f"| {facet} | {_format_metric(stats.get('precision'))} | {_format_metric(stats.get('recall'))} | "
            f"{_format_metric(stats.get('f1'))} | {stats.get('reviewed', 0)} | {stats.get('tp', 0)} | {stats.get('fp', 0)} | {stats.get('fn', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Review Coverage",
            f"- Unreviewed predicted positives: {report['unreviewed_predictions'].get('total', 0)}",
            f"- Decisions lacking candidate rows: {report['decisions_missing_candidates'].get('total', 0)}",
        ]
    )
    if "baseline_comparison" in report:
        base = report["baseline_comparison"]
        lines.extend(
            [
                "",
                "## Baseline Comparison",
                f"- Baseline run: {base.get('baseline_run_id')}",
                f"- Precision delta: {_format_metric(base.get('precision_delta'))}",
                f"- Recall delta: {_format_metric(base.get('recall_delta'))}",
            ]
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate IFC ↔ Uniclass candidate run against reviewer decisions")
    parser.add_argument("--config", default="config/settings.yaml", help="Path to settings YAML")
    parser.add_argument("--run-id", help="Candidate run UUID to evaluate")
    parser.add_argument("--baseline-run-id", help="Baseline run UUID for regression comparison")
    parser.add_argument("--accept-threshold", type=float, help="Override acceptance threshold for evaluation")
    parser.add_argument("--fail-on-regression", action="store_true", help="Exit non-zero when regression detected")
    parser.add_argument("--output", help="Optional explicit output path for JSON report")
    args = parser.parse_args()

    settings = load_settings(Path(args.config))
    conn = connect(settings.db_url)

    run_id = args.run_id or _latest_run_id(conn)
    if not run_id:
        raise SystemExit("No candidate history available; run candidate generation first.")

    threshold = args.accept_threshold or settings.evaluation_accept_threshold or settings.auto_accept
    baseline_run_id = args.baseline_run_id or settings.evaluation_baseline_run_id
    fail_on_regression = args.fail_on_regression or settings.evaluation_fail_on_regression

    current = evaluate_run(conn, run_id, threshold)
    baseline = evaluate_run(conn, baseline_run_id, threshold) if baseline_run_id else None

    report = build_report(current, baseline, threshold)

    out_path = Path(args.output) if args.output else settings.output_dir / "evaluations" / f"{run_id}.json"
    write_outputs(report, out_path)

    print(f"Run {run_id} evaluated. Precision={_format_metric(report['overall'].get('precision'))}, "
          f"Recall={_format_metric(report['overall'].get('recall'))}. Report saved to {out_path}")

    regression = False
    reasons = []
    overall = report.get("overall", {})
    curr_prec = overall.get("precision")
    curr_rec = overall.get("recall")
    if baseline:
        base = report["baseline_comparison"]
        delta_prec = base.get("precision_delta")
        if delta_prec is not None and delta_prec < -settings.evaluation_max_precision_drop:
            regression = True
            reasons.append(f"precision dropped by {delta_prec:.3f}")
        delta_rec = base.get("recall_delta")
        if delta_rec is not None and delta_rec < -settings.evaluation_max_precision_drop:
            regression = True
            reasons.append(f"recall dropped by {delta_rec:.3f}")
    if curr_rec is not None and curr_rec < settings.evaluation_min_recall:
        regression = True
        reasons.append(f"recall {curr_rec:.3f} below minimum {settings.evaluation_min_recall:.3f}")

    if fail_on_regression and regression:
        print("Regression detected: " + "; ".join(reasons))
        return 1
    if regression:
        print("Regression detected but not failing build: " + "; ".join(reasons))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
