
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import norm, skew, kurtosis
from shiny import App, reactive, render, ui
from shiny.types import SilentException

# ------------------------------------------------------------------ settings
RED, BLACK, GREY, LIGHT = "#c8102e", "#111111", "#9a9a9a", "#c4c4c4"
REGIME_COLOR = {"Bull": BLACK, "Chop": LIGHT, "Bear": RED}
STOCK_COLOR = {"ITC": BLACK, "MARUTI": RED, "TCS": GREY}
STOCKS = ["ITC", "MARUTI", "TCS"]
STATES = ["Bull", "Bear", "Chop"]
TEAM = [("Sherin Susan David", "25060242127"), ("Teerth Narang", "25060242153")]

# the final model
FINAL = dict(p_trend=0.95, p_chop=0.92, mom=1.0, chop_er=0.12, trend_er=0.30, cost=0.10)

# find the CSVs whether this file sits in HMM-stock-market/ or HMM-stock-market/dashboard/
HERE = os.path.dirname(os.path.abspath(__file__))
for cand in [os.path.join(HERE, "..", "Data_extractor"), os.path.join(HERE, "Data_extractor")]:
    if os.path.exists(os.path.join(cand, "ITC.NS.csv")):
        CSV_DIR = os.path.abspath(cand)
        break
else:
    raise FileNotFoundError("Can't find ITC.NS.csv. Run Data_extractor/extract.py first.")

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": LIGHT, "axes.grid": True, "grid.color": "#eeeeee",
                     "axes.titleweight": "bold", "axes.titlesize": 10, "axes.titlelocation": "left",
                     "legend.frameon": False, "figure.facecolor": "white"})


# ------------------------------------------------------------------ data
def load():
    data = {}
    for sym in STOCKS:
        df = pd.read_csv(os.path.join(CSV_DIR, f"{sym}.NS.csv"))
        df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None).dt.normalize()
        df = df.dropna(subset=["obs_mom", "obs_vol", "obs_er"]).reset_index(drop=True)
        df["ret"] = np.log(df.Close).diff()
        data[sym] = df
    return data


DATA = load()
DMIN = min(d.Date.min() for d in DATA.values()).date()
DMAX = max(d.Date.max() for d in DATA.values()).date()


# ------------------------------------------------------------------ model
def transition(pb, pr, pc):
    return np.array([
        [pb, (1 - pb) * 0.2, (1 - pb) * 0.8],
        [(1 - pr) * 0.2, pr, (1 - pr) * 0.8],
        [(1 - pc) * 0.5, (1 - pc) * 0.5, pc],
    ])


def forward_filter(L, A, start=None):
    post = np.full(L.shape[1], 1 / L.shape[1]) if start is None else start.copy()
    out = []
    for like in L:
        post = like * (A.T @ post)
        post /= post.sum()
        out.append(post.copy())
    return np.array(out)


def run_model(df, s):
    emis = {
        "Bull": {"mom": (s["mom"], 0.8), "vol": (-0.3, 1.0), "er": (s["trend_er"], 0.15)},
        "Bear": {"mom": (-s["mom"], 0.8), "vol": (0.5, 1.0), "er": (s["trend_er"], 0.15)},
        "Chop": {"mom": (0.0, 0.6), "vol": (0.0, 1.0), "er": (s["chop_er"], 0.08)},
    }
    L = np.column_stack([
        norm.pdf(df.obs_mom.clip(-4, 4), *emis[k]["mom"])
        * norm.pdf(df.obs_vol.clip(-4, 4), *emis[k]["vol"])
        * norm.pdf(df.obs_er, *emis[k]["er"]) for k in STATES])
    P = pd.DataFrame(forward_filter(L, transition(s["p_trend"], s["p_trend"], s["p_chop"])), columns=STATES, index=df.index)
    out = pd.concat([df, P], axis=1)
    out["regime"] = P.idxmax(axis=1)
    pos = (out.regime == "Bull").astype(float).shift(1).fillna(0)       # act the day after the signal
    out["pos"] = pos
    out["strat"] = pos * out.ret - pos.diff().abs().fillna(0) * s["cost"] / 100
    return out


def perf(x):
    x = x.dropna()
    if len(x) < 2:
        return np.nan, np.nan, np.nan
    w = np.exp(x.cumsum())
    return x.mean() * 252, x.mean() / x.std() * np.sqrt(252), (w / w.cummax() - 1).min()




def regime_figure(d):
    fig, (a1, a2) = plt.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [1.2, 1]})
    a1.plot(d.Date, d.Close, color=BLACK, lw=1.2)
    a1.set_ylabel("Close")
    a2.stackplot(d.Date, d.Bull, d.Chop, d.Bear, colors=[REGIME_COLOR[k] for k in ["Bull", "Chop", "Bear"]],
                 labels=["Bull", "Chop", "Bear"], lw=0)
    a2.set_ylim(0, 1)
    a2.set_ylabel("Probability")
    a2.grid(False)
    a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.tight_layout()
    return fig


def forward_table(d):
    lp = np.log(d.Close)
    nxt20 = lp.shift(-20) - lp
    rows = []
    for k in ["Bull", "Chop", "Bear"]:
        m = d.regime == k
        rows.append({"Regime": k, "Avg. next day": f"{d.ret.shift(-1)[m].mean():+.2%}",
                     "Avg. next 20 days": f"{nxt20[m].mean():+.2%}", "Days": int(m.sum())})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ any ticker: same pipeline as extract.py
def build_features(raw):
    df = raw[raw["Volume"] != 0].reset_index()
    ret = np.log(df["Close"]).diff()
    drift = ret.ewm(span=40, adjust=False).mean()
    rvol = ret.ewm(span=40, adjust=False).std()
    df["obs_mom"] = drift / rvol * np.sqrt(40)                                   # momentum: 40-day trend t-stat
    prev = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - prev).abs(), (df["Low"] - prev).abs()], axis=1).max(axis=1)
    atr_pct = np.log(tr.ewm(alpha=1 / 20, adjust=False).mean() / df["Close"])
    df["obs_vol"] = (atr_pct - atr_pct.rolling(250).mean()) / atr_pct.rolling(250).std()   # volatility z over 1 year
    net = (df["Close"] - df["Close"].shift(20)).abs()
    path = df["Close"].diff().abs().rolling(20).sum()
    df["obs_er"] = net / path                                                    # efficiency ratio, 20 days
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["obs_mom", "obs_vol", "obs_er"]).reset_index(drop=True)
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.normalize()
    df["ret"] = np.log(df.Close).diff()
    return df


def fetch(ticker, period):
    import yfinance as yf
    t = ticker.strip().upper()
    tries = [t] if ("." in t or t.startswith("^")) else [t + ".NS", t]
    for sym in tries:
        raw = yf.Ticker(sym).history(period=period, auto_adjust=True)
        if raw is not None and len(raw) > 0:
            raw.index.name = "Date"
            return sym, raw
    raise ValueError(f"No price data found for '{ticker}'. Check the ticker on finance.yahoo.com (NSE stocks end in .NS).")


# ------------------------------------------------------------------ accuracy
LABEL_MAP = {"Bull": "Up", "Chop": "Sideways", "Bear": "Down"}
CLASSES = ["Up", "Sideways", "Down"]


def outcome(move, thr):
    return np.where(move > thr, "Up", np.where(move < -thr, "Down", "Sideways"))


def accuracy_frame(d, h, k=0.5):
    lp = np.log(d.Close)
    fwd = lp.shift(-h) - lp
    past = lp - lp.shift(h)
    thr = k * d.ret.rolling(60).std() * np.sqrt(h)
    ok = fwd.notna() & thr.notna() & past.notna()
    a = pd.DataFrame({"Date": d.Date, "regime": d.regime, "fwd": fwd, "thr": thr, "past": past})[ok].copy()
    a["predicted"] = a.regime.map(LABEL_MAP)
    a["actual"] = outcome(a.fwd, a.thr)
    a["naive"] = outcome(a.past, a.thr)            
    return a


def accuracy_summary(a):
    if len(a) == 0:
        return {}
    acc = (a.predicted == a.actual).mean()
    naive = (a.naive == a.actual).mean()
    majority = a.actual.value_counts(normalize=True).max()
    chance = sum(a.predicted.value_counts(normalize=True).get(c, 0) * a.actual.value_counts(normalize=True).get(c, 0) for c in CLASSES)
    up_all, dn_all = (a.fwd > 0).mean(), (a.fwd < 0).mean()
    bull, bear = a[a.regime == "Bull"], a[a.regime == "Bear"]
    return dict(n=len(a), acc=acc, naive=naive, majority=majority, chance=chance,
                bull_hit=(bull.fwd > 0).mean() if len(bull) else np.nan, up_all=up_all,
                bear_hit=(bear.fwd < 0).mean() if len(bear) else np.nan, dn_all=dn_all)


# ------------------------------------------------------------------ UI
CSS = f"""
body {{ background: #fff; color: {BLACK}; }}
.navbar {{ border-bottom: 3px solid {RED} !important; background: #fff !important; }}
.navbar-brand {{ font-weight: 700; }}
.nav-link.active {{ color: {RED} !important; border-bottom: 2px solid {RED}; }}
.bslib-value-box .value-box-value {{ color: {RED}; }}
.card {{ border-top: 3px solid {BLACK}; }}
.form-range::-webkit-slider-thumb {{ background: {RED}; }}
.btn-primary, .btn-primary:hover, .btn-primary:focus {{ background: {RED} !important; border-color: {RED} !important; color: #fff !important; }}
h2.hero span {{ color: {RED}; }}
.irs--shiny .irs-bar, .irs--shiny .irs-single {{ background: {RED}; border-color: {RED}; }}
.irs--shiny .irs-handle {{ border-color: {RED}; }}
"""

sidebar = ui.sidebar(
    ui.input_select("stock", "Stock", {s: s for s in STOCKS}, selected="ITC"),
    ui.input_date_range("dates", "Date range", start=DMIN, end=DMAX, min=DMIN, max=DMAX),
    ui.hr(),
    ui.tags.b("Final model"),
    ui.tags.ul(
        ui.tags.li("Bull/Bear stay: 0.95, Chop stay: 0.92"),
        ui.tags.li("Momentum: 40-day trend t-stat"),
        ui.tags.li("Volatility: ATR % z-scored over 1 year"),
        ui.tags.li("Efficiency ratio: 20 days"),
        ui.tags.li("Cost: 0.1% per trade"),
        style="font-size:.85rem; padding-left:1.1rem;",
    ),
    width=320,
)

app_ui = ui.page_navbar(
    ui.nav_panel(
        "Overview",
        ui.h2(ui.HTML("Market <span>Regime</span> Engine"), class_="hero"),
        ui.p("A hidden Markov model that labels each trading day of an NSE stock as Bull, Bear or Chop (sideways), "
             "using only past prices. Pick a stock and dates on the left and every tab updates."),
        ui.layout_column_wrap(
            ui.value_box("Worst loss, buy & hold", ui.output_text("vb_bh_dd")),
            ui.value_box("Worst loss, Bull-only rule", ui.output_text("vb_bull_dd")),
            ui.value_box("Worst loss cut by", ui.output_text("vb_cut")),
            ui.value_box("Average regime length", ui.output_text("vb_run")),
            width=1 / 4,
        ),
        ui.card(ui.card_header("All three stocks, indexed to 100"), ui.output_plot("plot_index", height="320px")),
        ui.card(ui.card_header("Team"), ui.output_table("team")),
    ),
    ui.nav_panel(
        "Regime explorer",
        ui.card(ui.card_header(ui.output_text("regime_title")), ui.output_plot("plot_regime", height="520px")),
        ui.layout_columns(
            ui.card(ui.card_header("Share of days in each regime"), ui.output_table("tbl_share")),
            ui.card(ui.card_header("Average return after each regime"), ui.output_table("tbl_forward")),
        ),
    ),
    ui.nav_panel(
        "Backtest",
        ui.card(ui.card_header("Growth of ₹1: buy & hold vs holding only in Bull"), ui.output_plot("plot_equity", height="340px")),
        ui.card(ui.card_header("All stocks, selected dates and settings"), ui.output_table("tbl_backtest")),
    ),
    ui.nav_panel(
        "Data & features",
        ui.layout_columns(
            ui.card(ui.card_header("Feature distributions"),
                    ui.input_radio_buttons("feat", None, {"obs_mom": "Momentum", "obs_vol": "Volatility", "obs_er": "Efficiency"}, inline=True),
                    ui.output_plot("plot_hist", height="300px")),
            ui.card(ui.card_header("Daily return statistics"), ui.output_table("tbl_stats")),
        ),
    ),
    ui.nav_panel(
        "Final model",
        ui.card(ui.card_header("Final model parameters"), ui.output_table("tbl_params")),
        ui.card(ui.card_header("How it works"), ui.markdown(
            "Each day the model looks at three features built from past prices only: **momentum** (40-day trend t-statistic), "
            "**volatility** (20-day ATR as % of price, z-scored over one year) and the **efficiency ratio** (net 20-day move ÷ sum of daily moves). "
            "A forward filter carries yesterday's regime probabilities through the transition matrix, multiplies by how likely today's "
            "features are under each regime, and normalises. The most likely regime is the day's label.")),
    ),
    ui.nav_panel(
        "Any stock",
        ui.p("Type any Yahoo Finance ticker. The app downloads its prices, builds the same three features and runs the final model, "
             "whose settings were calibrated on ITC, Maruti and TCS. NSE stocks end in .NS (a bare name like RELIANCE is tried as RELIANCE.NS); "
             "US stocks need no suffix (AAPL); indices start with ^ (^NSEI)."),
        ui.layout_columns(
            ui.input_text("ticker", "Ticker", value="RELIANCE.NS"),
            ui.input_select("period", "History", {"2y": "2 years", "5y": "5 years", "10y": "10 years", "max": "All available"}, selected="5y"),
            ui.div(ui.input_action_button("go", "Run the model", class_="btn-primary"), style="padding-top:1.9rem"),
            col_widths=[4, 4, 4],
        ),
        ui.output_ui("any_status"),
        ui.layout_column_wrap(
            ui.value_box("Regime today", ui.output_text("any_now")),
            ui.value_box("Probability of that regime", ui.output_text("any_prob")),
            ui.value_box("In this regime for", ui.output_text("any_len")),
            ui.value_box("Worst loss cut by (Bull-only)", ui.output_text("any_cut")),
            width=1 / 4,
        ),
        ui.card(ui.card_header(ui.output_text("any_title")), ui.output_plot("any_regime", height="520px")),
        ui.layout_columns(
            ui.card(ui.card_header("Growth of 1 unit: buy & hold vs holding only in Bull"), ui.output_plot("any_equity", height="300px")),
            ui.card(ui.card_header("Regime summary and backtest"), ui.output_table("any_table"), ui.output_table("any_forward")),
        ),
    ),
    ui.nav_panel(
        "Accuracy",
        ui.markdown(
            "**How accuracy is measured.** A regime is a prediction about the next few weeks: Bull says *up*, Bear says *down*, Chop says *sideways*. "
            "For every day we check what the stock actually did over the next *h* trading days. The move counts as up or down only if it beats "
            "half of a normal *h*-day move for that stock (0.5 × 60-day daily volatility × √h); anything smaller is sideways. "
            "We compare the model with three baselines: **momentum** (assume the last *h* days continue), **always guess the most common outcome**, "
            "and **random guessing** with the model's own mix of labels."),
        ui.layout_columns(
            ui.input_select("acc_src", "Stock", {s: s for s in STOCKS}),
            ui.input_select("acc_h", "Horizon (trading days ahead)", {"5": "5 days", "10": "10 days", "20": "20 days", "60": "60 days"}, selected="20"),
            col_widths=[6, 6],
        ),
        ui.layout_column_wrap(
            ui.value_box("Model accuracy", ui.output_text("acc_model")),
            ui.value_box("Momentum baseline", ui.output_text("acc_naive")),
            ui.value_box("Always most common", ui.output_text("acc_major")),
            ui.value_box("Random guessing", ui.output_text("acc_chance")),
            width=1 / 4,
        ),
        ui.layout_columns(
            ui.card(ui.card_header("Predicted vs actual (share of days)"), ui.output_plot("acc_confusion", height="330px")),
            ui.card(ui.card_header("Direction hit rates"), ui.output_table("acc_hits"),
                    ui.p("A Bull hit rate above the 'any day' rate means Bull days were followed by gains more often than usual.", class_="text-muted small")),
        ),
        ui.card(ui.card_header("All stocks at this horizon"), ui.output_table("acc_all"),
                ui.p("Training stocks are in-sample: the model's settings were tuned on them, so these numbers are optimistic. "
                     "A ticker run in the Any stock tab is a fair out-of-sample test. Consecutive days share most of their future window, "
                     "so the effective sample is much smaller than the day count.", class_="text-muted small")),
    ),
    title="Market Regime Engine",
    sidebar=sidebar,
    header=ui.tags.style(CSS),
    fillable=False,
)


# ------------------------------------------------------------------ server
def server(input, output, session):

    @reactive.calc
    def modelled():
        """Model runs on the full history (so the filter has its past), then the dates are cut."""
        s = FINAL
        lo, hi = pd.Timestamp(input.dates()[0]), pd.Timestamp(input.dates()[1])
        out = {}
        for sym, df in DATA.items():
            m = run_model(df, s)
            out[sym] = m[(m.Date >= lo) & (m.Date <= hi)].reset_index(drop=True)
        return out

    @reactive.calc
    def cur():
        return modelled()[input.stock()]

    # ---- overview
    @reactive.calc
    def bt_cur():
        d = cur()
        return perf(d.ret), perf(d.strat)

    @render.text
    def vb_bh_dd():
        return f"{bt_cur()[0][2]:.0%}"

    @render.text
    def vb_bull_dd():
        return f"{bt_cur()[1][2]:.0%}"

    @render.text
    def vb_cut():
        b, q = bt_cur()
        return f"{1 - q[2] / b[2]:.0%}" if b[2] else "—"

    @render.text
    def vb_run():
        d = cur()
        sw = int((d.regime != d.regime.shift()).sum() - 1)
        return f"{len(d) / (sw + 1):.0f} days"

    @render.plot
    def plot_index():
        fig, ax = plt.subplots()
        for sym, d in modelled().items():
            if len(d):
                ax.plot(d.Date, d.Close / d.Close.iloc[0] * 100, color=STOCK_COLOR[sym], lw=1.6, label=sym)
        ax.axhline(100, color=GREY, ls="--", lw=0.8)
        ax.set_ylabel("Index (start = 100)")
        ax.legend(loc="upper left", ncol=3)
        return fig

    @render.table
    def team():
        return pd.DataFrame(TEAM, columns=["Name", "Roll number"])

    # ---- regime explorer
    @render.text
    def regime_title():
        return f"{input.stock()}: closing price and regime probabilities"

    @render.plot
    def plot_regime():
        return regime_figure(cur())

    @render.table
    def tbl_share():
        rows = []
        for sym, d in modelled().items():
            sh = d.regime.value_counts(normalize=True)
            sw = int((d.regime != d.regime.shift()).sum() - 1)
            rows.append({"Stock": sym, **{k: f"{sh.get(k, 0):.0%}" for k in ["Bull", "Chop", "Bear"]},
                         "Switches": sw, "Avg. length (days)": f"{len(d) / (sw + 1):.1f}"})
        return pd.DataFrame(rows)

    @render.table
    def tbl_forward():
        return forward_table(cur())

    # ---- backtest
    @render.plot
    def plot_equity():
        d = cur()
        fig, ax = plt.subplots()
        ax.plot(d.Date, np.exp(d.ret.fillna(0).cumsum()), color=GREY, lw=1.6, label="Buy & hold")
        ax.plot(d.Date, np.exp(d.strat.fillna(0).cumsum()), color=RED, lw=1.8, label="Hold only in Bull")
        ax.axhline(1, color=LIGHT, ls="--", lw=0.8)
        ax.set_ylabel("Growth of ₹1")
        ax.set_title(f"{input.stock()} (cost 0.1% per trade)")
        ax.legend(loc="upper left")
        return fig

    @render.table
    def tbl_backtest():
        rows = []
        for sym, d in modelled().items():
            b, q = perf(d.ret), perf(d.strat)
            rows.append({"Stock": sym, "B&H / yr": f"{b[0]:+.1%}", "Bull-only / yr": f"{q[0]:+.1%}",
                         "B&H Sharpe": f"{b[1]:.2f}", "Bull-only Sharpe": f"{q[1]:.2f}",
                         "B&H worst loss": f"{b[2]:.0%}", "Bull-only worst loss": f"{q[2]:.0%}",
                         "Trades": int(d.pos.diff().abs().sum()), "Invested": f"{d.pos.mean():.0%}"})
        return pd.DataFrame(rows)

    # ---- data & features
    @render.plot
    def plot_hist():
        fig, ax = plt.subplots()
        for sym, d in modelled().items():
            ax.hist(d[input.feat()], bins=30, histtype="step", lw=1.6, color=STOCK_COLOR[sym], label=sym, density=True)
        ax.set_yticklabels([])
        ax.legend()
        return fig

    @render.table
    def tbl_stats():
        rows = []
        for sym, d in modelled().items():
            r = d.ret.dropna()
            rows.append({"Stock": sym, "Days": len(d), "Return / yr": f"{r.mean() * 252:+.1%}",
                         "Volatility / yr": f"{r.std() * np.sqrt(252):.1%}", "Skew": f"{skew(r):+.2f}",
                         "Excess kurtosis": f"{kurtosis(r):.1f}", "Worst day": f"{r.min():+.1%}", "Best day": f"{r.max():+.1%}"})
        return pd.DataFrame(rows)

    # ---- final model
    @render.table
    def tbl_params():
        f = FINAL
        return pd.DataFrame([
            {"Regime": "Bull", "Momentum mean (sd)": f"+{f['mom']:.1f} (0.8)", "Volatility mean (sd)": "-0.3 (1.0)", "Efficiency mean (sd)": f"{f['trend_er']:.2f} (0.15)", "Chance of staying": f"{f['p_trend']:.2f}"},
            {"Regime": "Bear", "Momentum mean (sd)": f"-{f['mom']:.1f} (0.8)", "Volatility mean (sd)": "+0.5 (1.0)", "Efficiency mean (sd)": f"{f['trend_er']:.2f} (0.15)", "Chance of staying": f"{f['p_trend']:.2f}"},
            {"Regime": "Chop", "Momentum mean (sd)": "0.0 (0.6)", "Volatility mean (sd)": "0.0 (1.0)", "Efficiency mean (sd)": f"{f['chop_er']:.2f} (0.08)", "Chance of staying": f"{f['p_chop']:.2f}"},
        ])

    # ---- any stock
    custom = reactive.value(None)          # (ticker, modelled frame) or None
    custom_err = reactive.value("")

    @reactive.effect
    @reactive.event(input.go)
    def _run_any():
        custom_err.set("")
        with ui.Progress() as p:
            p.set(message=f"Downloading {input.ticker().strip().upper()}…")
            try:
                sym, raw = fetch(input.ticker(), input.period())
                df = build_features(raw)
                if len(df) < 60:
                    raise ValueError(f"Only {len(df)} usable days for {sym} after the one-year warm-up. Choose a longer history.")
                p.set(message="Running the model…")
                custom.set((sym, run_model(df, FINAL)))
                ui.update_select("acc_src", choices={**{s: s for s in STOCKS}, "custom": f"{sym} (Any stock tab)"}, selected="custom")
            except Exception as e:
                custom.set(None)
                custom_err.set(str(e))

    def need_custom():
        c = custom()
        if c is None:
            raise SilentException()
        return c

    @render.ui
    def any_status():
        if custom_err():
            return ui.div(custom_err(), class_="alert alert-danger")
        c = custom()
        if c is None:
            return ui.div("Enter a ticker and press Run the model.", class_="text-muted")
        sym, d = c
        return ui.div(f"{sym}: {len(d):,} trading days from {d.Date.iloc[0]:%d %b %Y} to {d.Date.iloc[-1]:%d %b %Y}.", class_="text-muted")

    @render.text
    def any_now():
        return need_custom()[1].regime.iloc[-1]

    @render.text
    def any_prob():
        d = need_custom()[1]
        return f"{d[d.regime.iloc[-1]].iloc[-1]:.0%}"

    @render.text
    def any_len():
        r = need_custom()[1].regime
        run = (r[::-1] != r.iloc[-1]).to_numpy().argmax() if (r != r.iloc[-1]).any() else len(r)
        return f"{run} trading days"

    @render.text
    def any_cut():
        d = need_custom()[1]
        b, q = perf(d.ret), perf(d.strat)
        return f"{1 - q[2] / b[2]:.0%}" if b[2] else "—"

    @render.text
    def any_title():
        c = custom()
        return f"{c[0]}: closing price and regime probabilities" if c else "Run a ticker to see its regimes"

    @render.plot
    def any_regime():
        return regime_figure(need_custom()[1])

    @render.plot
    def any_equity():
        d = need_custom()[1]
        fig, ax = plt.subplots()
        ax.plot(d.Date, np.exp(d.ret.fillna(0).cumsum()), color=GREY, lw=1.6, label="Buy & hold")
        ax.plot(d.Date, np.exp(d.strat.fillna(0).cumsum()), color=RED, lw=1.8, label="Hold only in Bull")
        ax.axhline(1, color=LIGHT, ls="--", lw=0.8)
        ax.legend(loc="upper left")
        return fig

    @render.table
    def any_table():
        sym, d = need_custom()
        sh = d.regime.value_counts(normalize=True)
        sw = int((d.regime != d.regime.shift()).sum() - 1)
        b, q = perf(d.ret), perf(d.strat)
        return pd.DataFrame([
            {"Measure": "Share of days Bull / Chop / Bear", "Value": " / ".join(f"{sh.get(k, 0):.0%}" for k in ["Bull", "Chop", "Bear"])},
            {"Measure": "Average regime length", "Value": f"{len(d) / (sw + 1):.1f} days"},
            {"Measure": "Return per year: buy & hold / Bull-only", "Value": f"{b[0]:+.1%} / {q[0]:+.1%}"},
            {"Measure": "Worst loss: buy & hold / Bull-only", "Value": f"{b[2]:.0%} / {q[2]:.0%}"},
            {"Measure": "Time invested (Bull-only)", "Value": f"{d.pos.mean():.0%}"},
        ])

    @render.table
    def any_forward():
        return forward_table(need_custom()[1])

    # ---- accuracy
    @reactive.calc
    def acc_frame():
        h = int(input.acc_h())
        if input.acc_src() == "custom":
            d = need_custom()[1]
        else:
            d = run_model(DATA[input.acc_src()], FINAL)
        return accuracy_frame(d, h)

    @reactive.calc
    def acc_sum():
        return accuracy_summary(acc_frame())

    @render.text
    def acc_model():
        return f"{acc_sum()['acc']:.0%}"

    @render.text
    def acc_naive():
        return f"{acc_sum()['naive']:.0%}"

    @render.text
    def acc_major():
        return f"{acc_sum()['majority']:.0%}"

    @render.text
    def acc_chance():
        return f"{acc_sum()['chance']:.0%}"

    @render.plot
    def acc_confusion():
        a = acc_frame()
        m = pd.crosstab(a.predicted, a.actual, normalize="all").reindex(index=CLASSES, columns=CLASSES, fill_value=0)
        fig, ax = plt.subplots()
        from matplotlib.colors import LinearSegmentedColormap
        ax.imshow(m.values, cmap=LinearSegmentedColormap.from_list("wr", ["#ffffff", RED]), vmin=0, vmax=max(m.values.max(), 0.01))
        for i in range(3):
            for j in range(3):
                v = m.values[i, j]
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", color="white" if v > m.values.max() * 0.6 else BLACK,
                        fontweight="bold" if i == j else "normal")
        ax.set_xticks(range(3), CLASSES)
        ax.set_yticks(range(3), [f"{k} ({LABEL_MAP[k].lower()})" for k in ["Bull", "Chop", "Bear"]])
        ax.tick_params(length=0)
        ax.grid(False)
        ax.set_xlabel("What happened next")
        ax.set_ylabel("Model said")
        fig.tight_layout()
        return fig

    @render.table
    def acc_hits():
        s_ = acc_sum()
        return pd.DataFrame([
            {"Check": "Up after a Bull day", "Model": f"{s_['bull_hit']:.0%}", "Any day": f"{s_['up_all']:.0%}"},
            {"Check": "Down after a Bear day", "Model": f"{s_['bear_hit']:.0%}", "Any day": f"{s_['dn_all']:.0%}"},
            {"Check": "Days tested", "Model": f"{s_['n']:,}", "Any day": ""},
        ])

    @render.table
    def acc_all():
        h = int(input.acc_h())
        frames = {s_: run_model(DATA[s_], FINAL) for s_ in STOCKS}
        c = custom()
        if c is not None:
            frames[f"{c[0]} (out of sample)"] = c[1]
        rows = []
        for name, d in frames.items():
            r = accuracy_summary(accuracy_frame(d, h))
            rows.append({"Stock": name, "Model": f"{r['acc']:.0%}", "Momentum": f"{r['naive']:.0%}",
                         "Most common": f"{r['majority']:.0%}", "Random": f"{r['chance']:.0%}",
                         "Up after Bull": f"{r['bull_hit']:.0%}", "Down after Bear": f"{r['bear_hit']:.0%}", "Days": f"{r['n']:,}"})
        return pd.DataFrame(rows)


app = App(app_ui, server)
