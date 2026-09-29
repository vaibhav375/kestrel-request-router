"""Champion vs challengers, following docs/CHALLENGER_PROTOCOL.md (fixed before this was run).

    python experiments/challengers.py            # all challengers, rolling-origin backtest -> out/challengers.md
    python experiments/challengers.py --no-bert  # skip the fine-tuned transformer (C6, C7)

Deviation from the protocol, recorded here and in the report: C5 uses scikit-learn's HistGradientBoostingClassifier
instead of LightGBM (LightGBM needs a system OpenMP library that is not on this machine; same histogram-GBDT method).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from scipy.stats import binomtest
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import kestrel.router as KR  # noqa: E402
from experiments.compare_models import build_frame  # noqa: E402

SEED = 0
FOLDS = [("Oct-Dec 2025", "2025-10-01", "2026-01-01"),
         ("Jan-Mar 2026", "2026-01-01", "2026-04-01"),
         ("Apr-Jun 2026", "2026-04-01", "2026-07-01")]
ALPHA = 50  # shrinkage towards the pooled vague distribution


# ------------------------------------------------------------------ champion and vague-prior variants
def champion(tr, va, min_support=None):
    old = KR.MIN_SUPPORT
    if min_support is not None:
        KR.MIN_SUPPORT = min_support
    try:
        p = KR.Router.fit(tr).predict(va)
    finally:
        KR.MIN_SUPPORT = old
    return p["team"].values, p["route_type"].values


def pooled_vague(tr):
    return tr.loc[tr["vague"], "final_team_cur"].value_counts().idxmax()


def shrunk_argmax(tr_v, va_keys, key_cols, prior: pd.Series):
    """argmax of (count(key, team) + ALPHA * prior(team)) / (n(key) + ALPHA)."""
    counts = tr_v.groupby(key_cols)["final_team_cur"].value_counts().unstack(fill_value=0)
    out = []
    for k in va_keys:
        c = counts.loc[k] if k in counts.index else pd.Series(0, index=prior.index)
        post = (c.reindex(prior.index, fill_value=0) + ALPHA * prior) / (c.sum() + ALPHA)
        out.append(post.idxmax())
    return np.array(out)


def vague_override(base, kinds, new):
    base = base.copy()
    m = kinds == "vague"
    base[m] = new[m]
    return base


def c1_product(tr, va, base, kinds):
    tv = tr[tr["vague"]]
    prior = tv["final_team_cur"].value_counts(normalize=True)
    return vague_override(base, kinds, shrunk_argmax(tv, va["product_family"].tolist(), "product_family", prior))


def c2_recent(tr, va, base, kinds):
    tv = tr[tr["vague"] & (tr["created_at"] >= tr["created_at"].max() - pd.DateOffset(months=6))]
    return vague_override(base, kinds, np.array([tv["final_team_cur"].value_counts().idxmax()] * len(va)))


def c3_product_warranty(tr, va, base, kinds):
    tv = tr[tr["vague"]]
    prior = tv["final_team_cur"].value_counts(normalize=True)
    keys = list(zip(va["product_family"], va["warranty_status"]))
    return vague_override(base, kinds, shrunk_argmax(tv, keys, ["product_family", "warranty_status"], prior))


# ------------------------------------------------------------------ learned challengers
def _text_feats(tr, va):
    vf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    vl = TfidfVectorizer(ngram_range=(1, 3), min_df=1, sublinear_tf=True)
    Xtr = hstack([vf.fit_transform(tr["norm"]), vl.fit_transform(tr["last"]) * 1.5]).tocsr()
    Xva = hstack([vf.transform(va["norm"]), vl.transform(va["last"]) * 1.5]).tocsr()
    return Xtr, Xva


def c5_gbdt(tr, va):
    Xtr, Xva = _text_feats(tr, va)
    svd = TruncatedSVD(256, random_state=SEED)
    Str, Sva = svd.fit_transform(Xtr), svd.transform(Xva)
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    cols = ["channel", "product_family", "warranty_status"]
    Mtr, Mva = ohe.fit_transform(tr[cols]), ohe.transform(va[cols])
    Ttr = np.c_[tr["created_at"].dt.hour, tr["created_at"].dt.dayofweek]
    Tva = np.c_[va["created_at"].dt.hour, va["created_at"].dt.dayofweek]
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, early_stopping=True,
                                         validation_fraction=0.1, random_state=SEED)
    clf.fit(np.hstack([Str, Mtr, Ttr]), tr["final_team_cur"])
    return clf.predict(np.hstack([Sva, Mva, Tva]))


def c8_noise_cleaned(tr, va):
    clear = tr[~tr["vague"]]
    Xtr, Xva = _text_feats(clear, va)
    y = clear["final_team_cur"].values
    lr = LogisticRegression(C=4.0, max_iter=3000)
    oof = cross_val_predict(lr, Xtr, y, cv=5, method="predict_proba")
    classes = np.unique(y)
    own = oof[np.arange(len(y)), np.searchsorted(classes, y)]
    keep = own >= 0.10
    lr.fit(Xtr[keep], y[keep])
    p = lr.predict(Xva)
    p[va["vague"].values] = pooled_vague(tr)
    return p, int((~keep).sum())


_EMB = {}


def embeddings(d):
    if "all" not in _EMB:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
        _EMB["all"] = pd.DataFrame(m.encode(d["text"].str.lower().tolist(), batch_size=128,
                                            normalize_embeddings=True, show_progress_bar=False), index=d.index)
    return _EMB["all"]


def c9_knn(tr, va, E):
    clear = tr[~tr["vague"]]
    knn = KNeighborsClassifier(n_neighbors=25, weights="distance", metric="cosine")
    knn.fit(E.loc[clear.index].values, clear["final_team_cur"])
    p = knn.predict(E.loc[va.index].values)
    p[va["vague"].values] = pooled_vague(tr)
    return p


def c6_bert(tr, va, epochs=3, max_len=64, bs=32, lr=5e-5):
    """Fine-tune all layers of bge-small (BERT, 33M params) as a 7-way classifier. Local, MPS if available."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    torch.manual_seed(SEED)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    name = "BAAI/bge-small-en-v1.5"
    classes = sorted(tr["final_team_cur"].unique())
    idx = {c: i for i, c in enumerate(classes)}
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=len(classes)).to(dev)
    enc = lambda s: tok(s.str.lower().tolist(), truncation=True, max_length=max_len, padding="max_length",
                        return_tensors="pt")
    Xtr = enc(tr["text"])
    ytr = torch.tensor([idx[c] for c in tr["final_team_cur"]])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * ((len(tr) + bs - 1) // bs)
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    g = torch.Generator().manual_seed(SEED)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(tr), generator=g)
        for i in range(0, len(tr), bs):
            b = perm[i:i + bs]
            out = model(input_ids=Xtr["input_ids"][b].to(dev), attention_mask=Xtr["attention_mask"][b].to(dev),
                        labels=ytr[b].to(dev))
            out.loss.backward()
            opt.step(); sch.step(); opt.zero_grad()
    model.eval()
    Xva = enc(va["text"])
    preds = []
    with torch.no_grad():
        for i in range(0, len(va), 128):
            logits = model(input_ids=Xva["input_ids"][i:i + 128].to(dev),
                           attention_mask=Xva["attention_mask"][i:i + 128].to(dev)).logits
            preds.extend(logits.argmax(-1).cpu().tolist())
    del model
    if dev == "mps":
        torch.mps.empty_cache()
    return np.array([classes[i] for i in preds])


# ------------------------------------------------------------------ harness
def run_fold(d, tr, va, with_bert, E):
    out, secs = {}, {}

    def timed(name, fn):
        t0 = time.time()
        out[name] = fn()
        secs[name] = time.time() - t0

    base, kinds = champion(tr, va)
    out["C0 champion"] = base
    timed("C1 vague prior by product", lambda: c1_product(tr, va, base, kinds))
    timed("C2 vague prior, last 6 months", lambda: c2_recent(tr, va, base, kinds))
    timed("C3 vague prior by product x warranty", lambda: c3_product_warranty(tr, va, base, kinds))
    timed("C4a MIN_SUPPORT=1", lambda: champion(tr, va, 1)[0])
    timed("C4b MIN_SUPPORT=10", lambda: champion(tr, va, 10)[0])
    timed("C5 gradient boosting (HGB)", lambda: c5_gbdt(tr, va))
    if with_bert:
        timed("C6 fine-tuned transformer", lambda: c6_bert(tr, va))
        pv = pooled_vague(tr)
        out["C7 transformer + vague override"] = np.where(va["vague"].values, pv, out["C6 fine-tuned transformer"])
    timed("C8 noise-cleaned TF-IDF+LR", lambda: c8_noise_cleaned(tr, va)[0])
    timed("C9 kNN on embeddings", lambda: c9_knn(tr, va, E))
    vote = pd.DataFrame({"a": out["C0 champion"], "b": out["C5 gradient boosting (HGB)"], "c": out["C9 kNN on embeddings"]})
    out["C10 majority vote (C0, C5, C9)"] = np.where(vote["b"] == vote["c"], vote["b"], vote["a"])
    return out, kinds, secs


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, adj, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    return adj


def main(with_bert=True):
    d = build_frame()
    E = embeddings(d)
    pooled, kinds_all, truth_all, fold_rows, secs_all = {}, [], [], [], {}
    for name, start, end in FOLDS:
        tr = d[d["created_at"] < start]
        va = d[(d["created_at"] >= start) & (d["created_at"] < end)]
        t0 = time.time()
        preds, kinds, secs = run_fold(d, tr, va, with_bert, E)
        truth = va["final_team_cur"].values
        for k, p in preds.items():
            pooled.setdefault(k, []).append(p)
            fold_rows.append({"fold": name, "model": k, "acc": float(np.mean(p == truth))})
        for k, s in secs.items():
            secs_all[k] = secs_all.get(k, 0) + s
        kinds_all.append(kinds)
        truth_all.append(truth)
        print(f"{name}: train {len(tr):,} test {len(va):,} ({time.time() - t0:.0f}s)", flush=True)

    truth = np.concatenate(truth_all)
    kinds = np.concatenate(kinds_all)
    P = {k: np.concatenate(v) for k, v in pooled.items()}
    champ_ok = P["C0 champion"] == truth
    rows, pvals = [], {}
    for k, p in P.items():
        ok = p == truth
        r = {"model": k, "pooled_acc": ok.mean(), "clear": ok[kinds != "vague"].mean(), "vague": ok[kinds == "vague"].mean(),
             "delta_pts": 100 * (ok.mean() - champ_ok.mean()), "secs": secs_all.get(k, np.nan)}
        if k != "C0 champion":
            b = int(np.sum(ok & ~champ_ok))   # challenger right, champion wrong
            c = int(np.sum(~ok & champ_ok))   # champion right, challenger wrong
            r["wins"], r["losses"] = b, c
            pvals[k] = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
        rows.append(r)
    adj = holm(pvals)
    res = pd.DataFrame(rows)
    res["p_mcnemar"] = res["model"].map(pvals)
    res["p_holm"] = res["model"].map(adj)
    folds = pd.DataFrame(fold_rows).pivot(index="model", columns="fold", values="acc")[[f[0] for f in FOLDS]]
    res = res.set_index("model").join(folds).sort_values("pooled_acc", ascending=False)
    res.to_csv(ROOT / "out" / "challengers.csv")
    pd.DataFrame({"truth": truth, "route_type": kinds, **P}).to_csv(ROOT / "out" / "challengers_predictions.csv", index=False)

    L = ["# Champion vs challengers: results", "",
         "Protocol and decision rule: `docs/CHALLENGER_PROTOCOL.md` (committed before this ran). "
         f"Rolling-origin backtest, {len(truth):,} pooled predictions, each quarter scored by models trained only on "
         "earlier data. Truth = closing team.", "",
         "Deviation: C5 uses scikit-learn HistGradientBoosting instead of LightGBM (LightGBM needs a system OpenMP "
         "library not present on this machine; same histogram-GBDT method).", "",
         "| model | pooled accuracy | vs champion | wins / losses vs champion | McNemar p | Holm p | clear | vague | "
         + " | ".join(f[0] for f in FOLDS) + " | fit+predict s (3 folds) |",
         "|---|---|---|---|---|---|---|---|" + "---|" * len(FOLDS) + "---|"]
    for k, r in res.iterrows():
        wl = "" if k == "C0 champion" else f"{int(r['wins'])} / {int(r['losses'])}"
        pm = "" if k == "C0 champion" else f"{r['p_mcnemar']:.3f}"
        ph = "" if k == "C0 champion" else f"{r['p_holm']:.3f}"
        sec = "" if pd.isna(r["secs"]) else f"{r['secs']:.0f}"
        L.append(f"| {'**' + k + '**' if k == 'C0 champion' else k} | {r['pooled_acc']:.2%} | {r['delta_pts']:+.2f} pts | {wl} | "
                 f"{pm} | {ph} | {r['clear']:.2%} | {r['vague']:.2%} | "
                 + " | ".join(f"{r[f[0]]:.2%}" for f in FOLDS) + f" | {sec} |")
    L.append("")
    better = res[(res.index != "C0 champion") & (res["delta_pts"] > 0)]
    sig = better[better["p_holm"] < 0.05]
    L += ["## Decision", ""]
    if sig.empty:
        L.append(f"No challenger beats the champion significantly (rule 2). {len(better)} scored higher in the pooled "
                 "backtest, none with Holm-corrected p < 0.05. **The champion stays.**")
    else:
        L.append("Significant winners (rules 3–4 checked in docs/CHALLENGER_RESULTS.md): " + ", ".join(sig.index))
    (ROOT / "out" / "challengers.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(with_bert="--no-bert" not in sys.argv)
