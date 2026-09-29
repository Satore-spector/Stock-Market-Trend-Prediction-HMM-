import numpy as np
import pandas as pd
from scipy.stats import norm
import os

#original Code that only picks up one mannually inputed csv at a time
"""
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
print(probs.idxmax(axis=1).value_counts())"""

#importing csv file
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
csv_dir = os.path.join(project_root, "Data_extractor")

data_frames = {}

for root, dir, files in os.walk(csv_dir):
    for filename in files:
        if filename.endswith(".csv"):
            file_path = os.path.join(root, filename)
            base_name = os.path.splitext(filename)[0]
            file_key = base_name[:3]
            data_frames[file_key] = pd.read_csv(file_path)
            print(f"imported as: {filename} -> {file_key}")

#dealing with dates

for key, i in data_frames.items():
    print(f"Converting dates for {key}")
    i['Date'] = pd.to_datetime(i['Date'], utc=True)
    print("Done")

for key, i in data_frames.items():
    print(f"Dataframes: {key}")
    print(i.head())
    print("\n")


p_bull, p_bear, p_chop = 0.80, 0.80, 0.60

states = ["Bull", "Bear", "Chop"]

A = np.array([
    [p_bull,               (1-p_bull)*0.2,       (1-p_bull)*0.8],
    [(1-p_bear)*0.2,        p_bear,               (1-p_bear)*0.8],
    [(1-p_chop)*0.5,       (1-p_chop)*0.5,         p_chop],
])
assert np.allclose(A.sum(axis=1), 1.0)

emis = {
    "Bull": {"mom": ( 1.0, 1.0), "atr": (-0.5, 1.0)},
    "Bear": {"mom": (-1.0, 1.0), "atr": ( 1.0, 1.0)},
    "Chop": {"mom": ( 0.0, 0.5), "atr": ( 1.5, 1.0)},
}

mom_dict = {}
atr_dict ={}

dicts = [mom_dict, atr_dict]

for key, i in data_frames.items():
    d = i.dropna(subset=['obs_mom', 'obs_vol'])
    mom_dict[f"{key}_mom"] = d['obs_mom'].clip(-3, 3)
    atr_dict[f"{key}_atr"] = d['obs_vol'].clip(-3, 3) 

for d in dicts:
    for key, i in d.items():
        print(f"Column from {key}")
        print(i.head(n=25))
        print("\n")

L_dict = {} #likelyhood matrix

for key in data_frames:   
    mom_series = mom_dict[f"{key}_mom"]
    atr_series = atr_dict[f"{key}_atr"]
    L = np.column_stack([
        norm.pdf(mom_series, *emis[s]["mom"]) * norm.pdf(atr_series, *emis[s]["atr"])
        for s in states
    ])
    
    # Save matrix to avoid overwriting L on the next loop
    L_dict[key] = L

print(L_dict[key][:5])

probs_dict = {}

for key in data_frames:
    L = L_dict[key]
    dates = data_frames[key]['Date']

    post = np.full(3, 1 / 3)
    out = []
    for like in L:
        post = like * (A.T @ post)
        post = post / post.sum()
        out.append(post)

    probs_dict[key] = pd.DataFrame(out, index=dates, columns=states)

for key, probs in probs_dict.items():
    print(key)
    print(probs.tail())
    print(probs.idxmax(axis=1).value_counts())
    print()

#checking our output

for key, probs in probs_dict.items():
    max_prob = probs.max(axis=1)
    print(key, "median confidence:", max_prob.median(), "| % of bars above 95%:", (max_prob > 0.95).mean())

for key, probs in probs_dict.items():
    dom = probs.idxmax(axis=1)
    switches = (dom != dom.shift()).sum() - 1
    print(key, "switches:", switches, "avg run:", len(dom) / (switches + 1))

#initial plot check
import matplotlib.pyplot as plt

for key, probs in probs_dict.items():
    df = data_frames[key]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    ax1.plot(df['Date'], df['Close'], color='black', linewidth=1)
    ax1.set_title(f"{key} — Close price")

    ax2.stackplot(probs.index, probs['Bull'], probs['Chop'], probs['Bear'],
                  labels=['Bull', 'Chop', 'Bear'],
                  colors=['#2ee319', '#7f49de', '#f31623'])
    ax2.legend(loc='upper left')
    ax2.set_title(f"{key} — Regime probability")

    plt.tight_layout()
    plt.savefig(f"{key}_regime_check.png")
    plt.show()

