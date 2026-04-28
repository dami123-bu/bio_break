from __future__ import annotations
import csv, json
from pathlib import Path
from typing import Any

def load_reports(paths: list[Path]) -> list[dict[str, Any]]:
    reports = []
    for p in paths:
        with p.open("r", encoding="utf-8") as f:
            reports.append(json.load(f))
    return reports

def summarize_report(report: dict[str, Any]) -> dict[str, Any]:
    m = report.get("metrics", {}) or {}
    return {
        "scenario": report.get("scenario"),
        "mode": report.get("mode"),
        "query": report.get("query"),
        "drug": report.get("drug"),
        "success": report.get("success", report.get("attack_success")),
        "partial_success": report.get("partial_success", False),
        "success_level": report.get("success_level"),
        "success_reason": report.get("success_reason"),
        "attack_in_top_k": m.get("attack_in_top_k"),
        "attack_top_1": m.get("attack_top_1"),
        "lowest_attack_rank": m.get("lowest_attack_rank"),
        "attack_hit_count": m.get("attack_hit_count"),
        "retrieval_bias": m.get("retrieval_bias"),
        "canary_seen": m.get("canary_seen"),
    }

def write_summary_csv(reports: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [summarize_report(r) for r in reports]
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def write_summary_md(reports: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [summarize_report(r) for r in reports]
    lines = ["# PharmaAttack Testbench Summary\n\n", "| Scenario | Mode | Success | Partial | Rank | Bias | Canary | Reason |\n", "|---|---|---:|---:|---:|---:|---:|---|\n"]
    for row in rows:
        vals = {k: ("" if v is None else v) for k, v in row.items()}
        lines.append(f"| {vals['scenario']} | {vals['mode']} | {vals['success']} | {vals['partial_success']} | {vals['lowest_attack_rank']} | {vals['retrieval_bias']} | {vals['canary_seen']} | {vals['success_reason']} |\n")
    output_path.write_text("".join(lines), encoding="utf-8")
