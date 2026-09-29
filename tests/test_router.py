from kestrel.data import TEAMS


def r(router, text):
    return router.route({"request_id": "T1", "request_text": text})


def test_seen_wording_uses_history(router):
    out = r(router, "good morning, motor of mixer grinder burnt smell")
    assert out["team"] == "Repairs"
    assert out["route_type"] == "seen"
    assert out["confidence_level"] == "high"
    assert not out["needs_clarification"]
    assert any("past requests with this wording" in x for x in out["reasons"])


def test_payment_remark_does_not_make_it_billing(router):
    out = r(router, "water purifier not turning on already paid in full")
    assert out["team"] == "Repairs"
    assert any("does not make it a Billing request" in x for x in out["reasons"])


def test_real_billing_request_stays_billing(router):
    out = r(router, "charged twice for my air fryer order, paid via card")
    assert out["team"] == "Billing"
    assert not any("does not make it a Billing request" in x for x in out["reasons"])


def test_last_request_decides(router):
    out = r(router, "how to clean air fryer, invoice not received for air fryer")
    assert out["team"] == "Billing"
    out = r(router, "invoice not received for air fryer, how to clean air fryer")
    assert out["team"] == "Product Advice"
    assert any("The last one decides" in x for x in out["reasons"])


def test_vague_request_is_flagged_with_a_question(router):
    out = r(router, "urgent: please call back regarding ceiling fan")
    assert out["route_type"] == "vague"
    assert out["needs_clarification"]
    assert out["team"] == "Repairs"  # most common outcome in the synthetic history
    q = out["clarifying_question"]
    assert {o["team"] for o in q["options"]} == set(TEAMS)


def test_new_wording_falls_back_to_keywords(router):
    out = r(router, "the installation guy never showed up for my new heater")
    assert out["route_type"] == "new_wording"
    assert out["team"] == "Installs & Demo"
    assert out["confidence_level"] == "medium"


def test_new_wording_matching_two_teams_asks(router):
    out = r(router, "the jar lid of my mixer cracked on day two")
    assert out["route_type"] == "new_wording"
    assert out["needs_clarification"]
    assert any("reads two ways" in x for x in out["reasons"])


def test_empty_and_garbage_text_do_not_crash(router):
    for text in ["", "   ", "???", "Ã©Ã©Ã©", "a" * 2000]:
        out = r(router, text)
        assert out["team"] in TEAMS
        assert isinstance(out["reasons"], list) and out["reasons"]


def test_every_answer_is_a_valid_team(router):
    for text in ["hello", "refund please", "my fan", "install", "warranty", "filter", "how do I use it"]:
        assert r(router, text)["team"] in TEAMS


def test_save_and_load_roundtrip(router, tmp_path):
    p = router.save(tmp_path / "m.joblib")
    loaded = type(router).load(p)
    text = "wrong model of air fryer delivered"
    assert loaded.route({"request_text": text})["team"] == router.route({"request_text": text})["team"]
