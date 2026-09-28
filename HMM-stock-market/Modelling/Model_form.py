import numpy as np
import pandas as pd
from scipy.stats import norm

ma = pd.read_csv(
    r"C:\GitHub projects\HMM-stock-market\Data_extractor\MARUTI.NS.csv",
    skiprows=range(1, 14),
)

# drop zero-volume bars (assign the result!)
ma = ma[ma["Volume"] != 0].reset_index(drop=True)

ma["Date"] = pd.to_datetime(ma["Date"], utc=True)

states = ["Bull", "Bear", "Chop"]

A = np.array([[0.90, 0.02, 0.08],
              [0.02, 0.90, 0.08],
              [0.10, 0.10, 0.80]])
assert np.allclose(A.sum(axis=1), 1.0)

emis = {
    #         (mean, std)     (mean, std)
    "Bull": {"mom": ( 0.8, 0.7), "atr": (-0.3, 0.8)},
    "Bear": {"mom": (-0.8, 0.7), "atr": ( 0.6, 1.0)},
    "Chop": {"mom": ( 0.0, 0.4), "atr": (-0.1, 0.8)},
}

mom = ma["sm_mom"].clip(-3, 3)
atr = ma["sm_ATR"].clip(-3, 3)

L = np.column_stack([
    norm.pdf(mom, *emis[s]["mom"]) * norm.pdf(atr, *emis[s]["atr"])
    for s in states
])

post = np.full(3, 1 / 3)        # uniform starting prior
out = []
for like in L:
    post = like * (A.T @ post)  # predict, then update
    post = post / post.sum()    # normalize
    out.append(post)

probs = pd.DataFrame(out, index=ma["Date"], columns=states)

print(probs.tail())
print(probs.idxmax(axis=1).value_counts())