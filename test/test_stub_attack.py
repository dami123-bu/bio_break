from pharma_attack.stub_attack import run_stub_keyword_hijack_demo

def test_stub_keyword_hijack_succeeds():
    report = run_stub_keyword_hijack_demo(query="What is tamoxifen used for?", drug="tamoxifen")
    assert report["attack_success"] is True
    assert "SECURITY_TEST_OVERRIDE" in report["attacked_output"]
