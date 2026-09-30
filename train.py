"""Train the router on all history and write predictions.csv for test_unlabelled.csv.

    python train.py                 # data pack in ./data
    python train.py --data other/   # a different folder

Writes:
  models/router.joblib                  the trained router (used by the service)
  predictions.csv                       request_id,team  - same rows and order as sample_submission.csv
  out/test_predictions_detailed.csv     plus confidence, route type, clarification flag
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from kestrel.data import DATA_DIR, TEAMS, _find, fix_mojibake, load_test, load_train  # noqa: E402
from kestrel.router import DEFAULT_MODEL_PATH, Router  # noqa: E402


CORRECTIONS_FILE = "corrections.csv"  # B5: written by POST /feedback, one row per agent-confirmed team


def load_corrections(data_dir: Path) -> pd.DataFrame:
    """B5: agent-confirmed teams for requests the router was unsure about, added to the next retrain."""
    path = data_dir / CORRECTIONS_FILE
    if not path.exists():
        return pd.DataFrame()
    c = pd.read_csv(path)
    c = c[c["team"].isin(TEAMS) & c["request_text"].notna()]
    return pd.DataFrame({
        "request_id": c.get("request_id"),
        "request_text": c["request_text"],
        "text": c["request_text"].map(fix_mojibake),
        "final_team_cur": c["team"],
        "product_family": c.get("product_family"),
        "created_at": pd.to_datetime(c["confirmed_at"]),
    })


def main(data_dir: Path) -> pd.DataFrame:
    t0 = time.time()
    train = load_train(data_dir)
    corrections = load_corrections(data_dir)
    if len(corrections):
        train = pd.concat([train, corrections], ignore_index=True)
        print(f"Added {len(corrections):,} agent-confirmed corrections from {CORRECTIONS_FILE}")
    router = Router.fit(train)
    router.save(DEFAULT_MODEL_PATH)
    print(f"Trained on {len(train):,} requests (until {router.trained_until}) in {time.time() - t0:.1f}s "
          f"-> {DEFAULT_MODEL_PATH.relative_to(ROOT)}")

    test = load_test(data_dir)
    detail = router.predict(test)
    sample = pd.read_csv(_find("sample_submission.csv", data_dir))
    pred = sample[["request_id"]].merge(detail[["request_id", "team"]], on="request_id", how="left")

    # Hard checks on the deliverable.
    assert len(pred) == len(sample) == len(test), "row count mismatch"
    assert pred["request_id"].is_unique, "duplicate request_id"
    assert set(pred["request_id"]) == set(test["request_id"]), "ids differ from test_unlabelled.csv"
    assert pred["team"].isin(TEAMS).all(), f"unknown team: {set(pred['team']) - set(TEAMS)}"

    pred.to_csv(ROOT / "predictions.csv", index=False)
    (ROOT / "out").mkdir(exist_ok=True)
    cols = ["request_id", "team", "confidence", "confidence_level", "route_type", "needs_clarification",
            "deciding_request"]
    detail[cols].to_csv(ROOT / "out" / "test_predictions_detailed.csv", index=False)

    print(f"predictions.csv: {len(pred):,} rows, all ids present once, all teams valid.")
    print(pred["team"].value_counts().to_string())
    print(detail["route_type"].value_counts(normalize=True).round(3).to_string())
    return detail


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA_DIR)
    main(ap.parse_args().data)
