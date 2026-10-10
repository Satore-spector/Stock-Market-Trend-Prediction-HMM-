"""
Fit the regime model's parameters from data (Baum-Welch via hmmlearn)
instead of setting them by hand, then compare with the hand-tuned model
on a period the fitted model has NOT seen.

Train: all data up to TRAIN_END.  Test: everything after.
Run from the repo folder:  python HMM-stock-market/Modelling/fit_hmm.py
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from hmmlearn.hmm import GaussianHMM

TRAIN_END = "2024-12-31"
FEATURES = ["obs_mom", "obs_vol", "obs_er"]
COST = 0.001

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
csv_dir = os.path.join(project_root, "Data_extractor")
out_dir = os.path.join(project_root, "results")
os.makedirs(out_dir, exist_ok=True)

# ---------------------------------------------------------------- load
data = {}
for f in sorted(os.listdir(csv_dir)):
    if f.endswith(".csv"):
        df = pd.read_csv(os.path.join(csv_dir, f))
        df["Date"] = pd.to_datetime(df["Date"], utc=True)
        df = df.dropna(subset=FEATURES).reset_index(drop=True)
        df["obs_mom"] = df["obs_mom"].clip(-4, 4)
        df["obs_vol"] = df["obs_vol"].clip(-4, 4)
        data[f.replace(".csv", "")] = df

# ---------------------------------------------------------------- fit on the training period, all stocks pooled
train = [d[d["Date"] <= pd.Timestamp(TRAIN_END, tz="UTC")] for d in data.values()]
X = np.vstack([t[FEATURES].values for t in train])
lengths = [len(t) for t in train]

best = None
for seed in range(10):                     # Baum-Welch can get stuck: keep the best of 10 starts
    m = GaussianHMM(n_components=3, covariance_type="full", n_iter=500, random_state=seed)
    m.fit(X, lengths)
    score = m.score(X, lengths)
    if best is None or score > best[0]:
        best = (score, m)
model = best[1]

# The fitted states have no names: lowest mean momentum = Bear, highest = Bull.
# NOTE: on this data the fitted states end up separated mainly by VOLATILITY
# (the "Chop" state is a calm uptrend), so the strategy below holds the stock
# in every state except Bear.
order = np.argsort(model.means_[:, 0])
names = {order[0]: "Bear", order[1]: "Chop", order[2]: "Bull"}
A = model.transmat_
means, covs = model.means_, model.covars_

print("Fitted parameters (training period)")
for k in order:
    print(f"  {names[k]:4s}  mean mom={means[k,0]:+.2f}  vol={means[k,1]:+.2f}  er={means[k,2]:.2f}"
          f"  | stays {A[k,k]:.3f}  (avg run ~{1/(1-A[k,k]):.0f} days)")


def forward_filter(X):
    """Same causal filter as Model_form.py, using the fitted parameters."""
    L = np.column_stack([multivariate_normal.pdf(X, means[k], covs[k]) for k in range(3)])
    post = model.startprob_.copy()
    out = []
    for like in L:
        post = like * (A.T @ post)
        post = post / post.sum()
        out.append(post)
    return np.array([names[k] for k in np.argmax(out, axis=1)])


def stats(daily):
    w = np.exp(daily.cumsum())
    return daily.mean() * 252 * 100, (w / w.cummax() - 1).min() * 100


# ---------------------------------------------------------------- test on unseen period
rows = []
for sym, df in data.items():
    df["regime"] = forward_filter(df[FEATURES].values)
    ret = np.log(df["Close"]).diff()
    pos = (df["regime"] != "Bear").astype(float).shift(1).fillna(0)
    strat = pos * ret - pos.diff().abs().fillna(0) * COST
    test = df["Date"] > pd.Timestamp(TRAIN_END, tz="UTC")

    bh_ret, bh_dd = stats(ret[test].dropna())
    s_ret, s_dd = stats(strat[test].dropna())
    share = df.loc[test, "regime"].value_counts(normalize=True)
    rows.append({"stock": sym,
                 "buy_hold_ann_%": bh_ret, "buy_hold_maxDD_%": bh_dd,
                 "fitted_not_bear_ann_%": s_ret, "fitted_not_bear_maxDD_%": s_dd,
                 "bull_%": share.get("Bull", 0) * 100, "chop_%": share.get("Chop", 0) * 100,
                 "bear_%": share.get("Bear", 0) * 100})
    df[["Date", "Close", "regime"]].to_csv(os.path.join(out_dir, f"{sym}_fitted_regimes.csv"), index=False)

res = pd.DataFrame(rows).round(1)
print(f"\nOut-of-sample test (after {TRAIN_END})\n", res.to_string(index=False))
res.to_csv(os.path.join(out_dir, "fitted_hmm_test.csv"), index=False)
