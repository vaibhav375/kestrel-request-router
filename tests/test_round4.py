"""Round 4 switches (docs/ROUND4_PROTOCOL.md), exercised on the synthetic history with every switch on."""
import pandas as pd
import pytest

from kestrel.repair import hinglish_to_english, spell_correct
from kestrel.router import ALL_FEATURES, Router
from tests.conftest import synthetic_history

PARA = pd.DataFrame({
    "source": ["invoice not received for P", "installer did not turn up for P"],
    "paraphrase": ["my bill never arrived for the water purifier", "the fitting guy never came for my water purifier"],
})


@pytest.fixture(scope="module")
def router_all():
    df = synthetic_history()
    df["product_family"] = ["Water Purifier" if i % 2 else "Ceiling Fan" for i in range(len(df))]
    return Router.fit(df, features=ALL_FEATURES, paraphrases=PARA)


def route(r, text, product="Water Purifier"):
    return r.route({"request_text": text, "product_family": product})


def test_hinglish_dictionary():
    assert hinglish_to_english("mera purifier kharab ho gaya") == "mera purifier not working"
    assert hinglish_to_english("fan se paani tapak raha hai") == "fan se leaking water"
    assert hinglish_to_english("bill nahi mila") == "invoice not received"


def test_spelling_correction_keeps_hindi_words():
    vocab = {"purifier", "leaking", "water", "bottom"}
    assert spell_correct("purifer leking watr", vocab) == "purifier leaking water"
    assert spell_correct("mera purifier", vocab) == "mera purifier"


def test_a1_vague_guess_is_labelled_as_a_guess(router_all):
    out = route(router_all, "please call back regarding water purifier")
    assert out["route_type"] == "vague" and out["needs_clarification"]
    assert any("This is a guess" in x for x in out["reasons"])


def test_b4_typos_are_repaired_to_known_wording(router_all):
    out = route(router_all, "motor of air fryer brnt smel")
    assert out["team"] == "Repairs"
    assert out["route_detail"] == "repaired"


def test_b4_hinglish_is_routed(router_all):
    assert route(router_all, "mera water purifier kharab ho gaya")["team"] == "Repairs"
    assert route(router_all, "air fryer ka bill nahi mila")["team"] == "Billing"


def test_b3_paraphrase_lookup(router_all):
    out = route(router_all, "my bill never arrived for the water purifier")
    assert out["team"] == "Billing" and out["route_detail"] == "paraphrase"


def test_b1_similar_wording(router_all):
    out = route(router_all, "wrong model of the air fryer was delivered to me")
    assert out["team"] == "Returns & Replacement"
    assert out["route_type"] in ("seen", "new_wording")


def test_c1_auto_route_flag(router_all):
    assert route(router_all, "motor of air fryer burnt smell")["auto_route"] is True
    assert route(router_all, "please call back regarding water purifier")["auto_route"] is False


def test_c2_three_likely_options(router_all):
    q = route(router_all, "issue with water purifier")["clarifying_question"]
    assert sum(o["likely"] for o in q["options"]) == 3
    assert len(q["options"]) == 7


def test_switches_off_matches_round3(router):
    # the session-wide `router` fixture is built with no switches: round-3 behaviour
    out = router.route({"request_text": "the installation guy never showed up for my new heater"})
    assert out["route_detail"] == "rules"
