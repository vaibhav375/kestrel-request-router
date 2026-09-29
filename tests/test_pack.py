"""Checks against the real data pack. Skipped when data/ is empty (the pack is confidential, policy §10)."""
import pandas as pd
import pytest

from kestrel.data import TEAMS, load_test, load_train

pytestmark = pytest.mark.pack


@pytest.fixture(scope="module")
def pack(pack_available):
    if not pack_available:
        pytest.skip("data pack not in data/")
    return load_train(), load_test()


def test_renamed_teams_are_merged(pack):
    d, _ = pack
    assert set(d["final_team_cur"]) == set(TEAMS)
    assert set(d["bot_team"]) == set(TEAMS)


def test_legacy_resolution_times_are_converted_from_utc(pack):
    d, _ = pack
    assert (d["hours_to_resolve"] >= 0).all()  # 26% of legacy rows are negative before the +5:30 fix
    med = d.groupby("source")["hours_to_resolve"].median()
    assert abs(med["crm"] - med["legacy_zoho"]) < 1.0


def test_no_mojibake_left(pack):
    d, _ = pack
    assert not d["text"].str.contains("Ã|Â|â€").any()


def test_predictions_file_matches_the_sample(pack):
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    pred_path = root / "predictions.csv"
    if not pred_path.exists():
        pytest.skip("run python train.py first")
    pred = pd.read_csv(pred_path)
    sample = pd.read_csv(next((root / "data").glob("*sample_submission.csv")))
    assert list(pred.columns) == list(sample.columns) == ["request_id", "team"]
    assert pred["request_id"].tolist() == sample["request_id"].tolist()
    assert pred["team"].isin(TEAMS).all()


def test_out_of_time_accuracy_does_not_regress(pack):
    from kestrel.router import Router
    d, _ = pack
    tr, va = d[d["created_at"] < "2026-04-01"], d[d["created_at"] >= "2026-04-01"]
    p = Router.fit(tr).predict(va)
    acc = (p["team"].values == va["final_team_cur"].values).mean()
    bot = (va["bot_team"].values == va["final_team_cur"].values).mean()
    assert acc >= 0.84, acc   # measured 85.7% on 30 Sep 2026
    assert acc - bot >= 0.07  # the bot scores 76.7% on the same requests
