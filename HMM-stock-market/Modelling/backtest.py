"""
Evaluate the regime engine.

1. What happened next: average return over the 1 and 20 trading days
   after each regime label.
2. Simple strategy: hold the stock while the regime is Bull, cash otherwise.
   The position is taken the day AFTER the signal (no look-ahead) and every
   change of position pays a transaction cost.

Keep the parameters below in sync with Model_form.py.
Run from the repo folder:  python HMM-stock-market/Modelling/backtest.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# ---------------------------------------------------------------- settings
COST = 0.001          # 0.1% per position change (brokerage + slippage)
FWD = 20              # forward horizon in trading days

states = ["Bull", "Bear", "Chop"]
p_bull, p_bear, p_chop = 0.95, 0.95, 0.92
A = np.array([
    [p_bull,             (1 - p_bull) * 0.2, (1 - p_bull) * 0.8],
    [(1 - p_bear) * 0.2, p_bear,             (1 - p_bear) * 0.8],
    [(1 - p_chop) * 0.5, (1 - p_chop) * 0.5, p_chop],
])
emis = {
    "Bull": {"mom": ( 1.0, 0.8), "atr": (-0.3, 1.0), "er": (0.30, 0.15)},
    "Bear": {"mom": (-1.0, 0.8), "atr": ( 0.5, 1.0), "er": (0.30, 0.15)},
    "Chop": {"mom": ( 0.0, 0.6), "atr": ( 0.0, 1.0), "er": (0.12, 0.08)},
}

# ---------------------------------------------------------------- paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
csv_dir = os.path.join(project_root, "Data_extractor")
out_dir = os.path.join(project_root, "results")
os.makedirs(out_dir, exist_ok=True)


def regimes(df):
    """Forward-filtered regime label for each day (uses only past data)."""
    L = np.column_stack([
        norm.pdf(df["obs_mom"].clip(-4, 4), *emis[s]["mom"])
        * norm.pdf(df["obs_vol"].clip(-4, 4), *emis[s]["atr"])
        * norm.pdf(df["obs_er"], *emis[s]["er"])
        for s in states
    ])
    post = np.full(3, 1 / 3)
    out = []
    for like in L:
        post = like * (A.T @ post)
        post = post / post.sum()
        out.append(post)
    return pd.Series(np.array(states)[np.argmax(out, axis=1)], index=df.index)


def stats(daily):
    ann_ret = daily.mean() * 252
    sharpe = daily.mean() / daily.std() * np.sqrt(252)
    wealth = np.exp(daily.cumsum())
    max_dd = (wealth / wealth.cummax() - 1).min()
    return ann_ret * 100, sharpe, max_dd * 100


rows, fwd_tables = [], []
for f in sorted(os.listdir(csv_dir)):
    if not f.endswith(".csv"):
        continue
    sym = f.replace(".csv", "")
    df = pd.read_csv(os.path.join(csv_dir, f))
    df["Date"] = pd.to_datetime(df["Date"], utc=True)
    df = df.dropna(subset=["obs_mom", "obs_vol", "obs_er"]).reset_index(drop=True)

    df["regime"] = regimes(df)
    logp = np.log(df["Close"])
    ret = logp.diff()

    # 1. what happened next
    t = pd.DataFrame({
        "regime": df["regime"],
        "next_1d_%": ret.shift(-1) * 100,
        f"next_{FWD}d_%": (logp.shift(-FWD) - logp) * 100,
    })
    fwd = t.groupby("regime").mean().round(2)
    fwd["days"] = t["regime"].value_counts()
    fwd.insert(0, "stock", sym)
    fwd_tables.append(fwd)

    # 2. strategy: in the stock during Bull, cash otherwise, acts next day
    pos = (df["regime"] == "Bull").astype(float).shift(1).fillna(0)
    trades = pos.diff().abs().fillna(0)
    strat = (pos * ret - trades * COST).dropna()
    bh = ret.dropna()

    b = stats(bh)
    s = stats(strat)
    rows.append({
        "stock": sym,
        "buy_hold_ann_%": b[0], "buy_hold_sharpe": b[1], "buy_hold_maxDD_%": b[2],
        "bull_only_ann_%": s[0], "bull_only_sharpe": s[1], "bull_only_maxDD_%": s[2],
        "trades": int(trades.sum()), "time_in_market_%": pos.mean() * 100,
    })

    # equity curve
    plt.figure(figsize=(10, 4))
    plt.plot(df["Date"].iloc[1:], np.exp(bh.cumsum()), label="Buy & hold", color="grey")
    plt.plot(df["Date"].iloc[1:], np.exp(strat.cumsum()), label="Hold only in Bull", color="#2ee319")
    plt.title(f"{sym} — growth of 1 rupee (after {COST*100:.1f}% costs)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{sym}_equity.png"))
    plt.close()

fwd_all = pd.concat(fwd_tables)
summary = pd.DataFrame(rows).round(2)

print("\nWhat happened after each regime\n", fwd_all)
print("\nStrategy vs buy & hold\n", summary.to_string(index=False))

fwd_all.to_csv(os.path.join(out_dir, "forward_returns.csv"))
summary.to_csv(os.path.join(out_dir, "backtest_summary.csv"), index=False)
print(f"\nSaved tables and charts to {out_dir}")
