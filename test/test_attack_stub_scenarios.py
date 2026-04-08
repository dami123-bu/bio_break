from pharma_help.attacks.payloads import CANARY_OVERRIDE
from pharma_help.attacks.stub_attack import run_stub_keyword_hijack_demo


def test_stub_keyword_hijack_promotes_attack_doc() -> None:
    report = run_stub_keyword_hijack_demo(
        query="What is tamoxifen used for?",
        drug="tamoxifen",
    )
    assert report["attack_success"] is True
    assert report["attacked"]["top_doc"]["id"].startswith("attack_stub_")
    assert CANARY_OVERRIDE in report["attacked"]["answer"]


def test_baseline_run_has_no_canary_in_answer() -> None:
    report = run_stub_keyword_hijack_demo(
        query="What is tamoxifen used for?",
        drug="tamoxifen",
    )
    assert report["baseline"]["canary_in_answer"] is False
