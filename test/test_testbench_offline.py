from pathlib import Path
from pharma_attack.testbench import run_testbench

def test_offline_testbench_writes_summary(tmp_path: Path):
    reports = run_testbench(mode="offline", output_dir=tmp_path, top_k=10)
    assert reports
    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "summary.md").exists()
    assert any(report.get("scenario") == "a1a_passive_rag_poison" for report in reports)
