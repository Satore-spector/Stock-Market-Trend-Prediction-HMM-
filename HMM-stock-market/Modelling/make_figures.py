"""
Figures 2-4 for the report, using exactly the same data handling and fit as fit_hmm.py.

  Figure 2: distribution of each feature per fitted hidden state
  Figure 3: heatmap of the fitted state transition matrix
  Figure 4: Baum-Welch log-likelihood convergence across the 10 restarts

Run from the repo folder:  python HMM-stock-market/Modelling/make_figures.py
Needs: numpy pandas scipy hmmlearn matplotlib seaborn
Outputs go to HMM-stock-market/results/.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal
from hmmlearn.hmm import GaussianHMM

TRAIN_END = "2024-12-31"
FEATURES = ["obs_mom", "obs_vol", "obs_er"]
N_RESTARTS = 10

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
csv_dir = os.path.join(project_root, "Data_extractor")
out_dir = os.path.join(project_root, "results")
os.makedirs(out_dir, exist_ok=True)

# ---------------------------------------------------------------- load (same as fit_hmm.py)
data = {}
for f in sorted(os.listdir(csv_dir)):
    if f.endswith(".csv"):
        df = pd.read_csv(os.path.join(csv_dir, f))
        df["Date"] = pd.to_datetime(df["Date"], utc=True)
        df = df.dropna(subset=FEATURES).reset_index(drop=True)
        df["obs_mom"] = df["obs_mom"].clip(-4, 4)
        df["obs_vol"] = df["obs_vol"].clip(-4, 4)
        data[f.replace(".csv", "")] = df

train = [d[d["Date"] <= pd.Timestamp(TRAIN_END, tz="UTC")] for d in data.values()]
X = np.vstack([t[FEATURES].values for t in train])
lengths = [len(t) for t in train]

# ---------------------------------------------------------------- fit: best of N restarts (same as fit_hmm.py)
best, curves = None, []
for seed in range(N_RESTARTS):
    m = GaussianHMM(n_components=3, covariance_type="full", n_iter=500, random_state=seed)
    m.fit(X, lengths)
    curves.append(list(m.monitor_.history))      # log-likelihood per EM iteration
    score = m.score(X, lengths)
    if best is None or score > best[0]:
        best = (score, m, seed)
_, model, best_seed = best

# Order states by mean momentum, as in fit_hmm.py, and use the report's labels
order = np.argsort(model.means_[:, 0])
labels = ["Down", "Calm uptrend", "Volatile uptrend"]
names = {order[i]: labels[i] for i in range(3)}
palette = dict(zip(labels, ["#d62728", "#2ca02c", "#ff7f0e"]))
A, means, covs = model.transmat_, model.means_, model.covars_

# ---------------------------------------------------------------- numbers to paste into the report
A_ord = A[np.ix_(order, order)]
print(f"Best restart: seed {best_seed}, training log-likelihood {best[0]:.2f}, n = {len(X)}")
print("\nFitted transition matrix (rows = from, cols = to):")
print(pd.DataFrame(A_ord, index=labels, columns=labels).round(3).to_string())
print("\nEmission means and average run length:")
for i, k in enumerate(order):
    print(f"  {labels[i]:17s} mom={means[k,0]:+.2f}  vol={means[k,1]:+.2f}  er={means[k,2]:.2f}"
          f"  | stays {A[k,k]:.3f}  (avg run ~{1/(1-A[k,k]):.0f} days)")
print("\nCovariance matrices (features: obs_mom, obs_vol, obs_er):")
for i, k in enumerate(order):
    print(f"  {labels[i]}\n{np.round(covs[k], 3)}")

# AIC / BIC across K (pooled training data), to fill the model-selection placeholder
T = len(X)
rows = []
for K in range(2, 7):
    mk = max((GaussianHMM(n_components=K, covariance_type="full", n_iter=500, random_state=s).fit(X, lengths)
              for s in range(N_RESTARTS)), key=lambda z: z.score(X, lengths))
    ll = mk.score(X, lengths)
    d = X.shape[1]
    p = (K - 1) + K * (K - 1) + K * d + K * d * (d + 1) // 2
    rows.append({"K": K, "params": p, "logL": ll, "AIC": 2 * p - 2 * ll, "BIC": p * np.log(T) - 2 * ll})
print("\nModel selection:\n", pd.DataFrame(rows).round(1).to_string(index=False))


# ---------------------------------------------------------------- causal forward filter (same as fit_hmm.py)
def forward_filter(Xs):
    L = np.column_stack([multivariate_normal.pdf(Xs, means[k], covs[k]) for k in range(3)])
    post, out = model.startprob_.copy(), []
    for like in L:
        post = like * (A.T @ post)
        post = post / post.sum()
        out.append(post)
    return np.array([names[k] for k in np.argmax(out, axis=1)])


# ---------------------------------------------------------------- Figure 2: feature distributions per state
frames = []
for t in train:
    tmp = t[FEATURES].copy()
    tmp["state"] = forward_filter(t[FEATURES].values)
    frames.append(tmp)
pooled = pd.concat(frames, ignore_index=True)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
titles = {"obs_mom": "Trend t-statistic (obs_mom)",
          "obs_vol": "Relative volatility (obs_vol)",
          "obs_er": "Efficiency ratio (obs_er)"}
for ax, feat in zip(axes, FEATURES):
    sns.kdeplot(data=pooled, x=feat, hue="state", hue_order=labels, palette=palette,
                common_norm=False, fill=True, alpha=0.25, linewidth=1.5, ax=ax, legend=(feat == "obs_er"))
    ax.set_title(titles[feat]); ax.set_xlabel(feat)
axes[0].set_ylabel("Density")
fig.suptitle("Feature distributions by hidden state (training period, forward-filtered labels)")
fig.tight_layout()
fig.savefig(os.path.join(out_dir, "figure_2_feature_distributions.png"), dpi=200)

# ---------------------------------------------------------------- Figure 3: transition-matrix heatmap
fig, ax = plt.subplots(figsize=(5.5, 4.5))
sns.heatmap(A_ord, annot=True, fmt=".3f", cmap="Blues", vmin=0, vmax=1,
            xticklabels=labels, yticklabels=labels, ax=ax)
ax.set_xlabel("To state"); ax.set_ylabel("From state")
plt.setp(ax.get_yticklabels(), rotation=0)
ax.set_title("Fitted state transition matrix")
fig.tight_layout()
fig.savefig(os.path.join(out_dir, "figure_3_transition_heatmap.png"), dpi=200)

# ---------------------------------------------------------------- Figure 4: convergence across restarts
fig, ax = plt.subplots(figsize=(7, 4.5))
for s, c in enumerate(curves):
    ax.plot(range(1, len(c) + 1), c, alpha=0.7, lw=2.2 if s == best_seed else 1,
            label=f"seed {s}" + (" (best)" if s == best_seed else ""))
ax.set_xlabel("EM iteration"); ax.set_ylabel("Log-likelihood")
ax.set_title("Baum-Welch convergence across random restarts")
ax.legend(fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, "figure_4_convergence.png"), dpi=200)

print("\nSaved figure_2_feature_distributions.png, figure_3_transition_heatmap.png, figure_4_convergence.png to", out_dir)
