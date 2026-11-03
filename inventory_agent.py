# %% [markdown]
# # Inventory Replenishment Agent
# **Course:** AAI 2025 · **Goal:** a daily agent that forecasts demand (EWMA), sizes safety stock from forecast
# error and a target service level, projects inventory over *lead time + review period*, and either places a
# purchase order (PO) or waits — then is simulated day by day and compared against a static baseline.
#
# **How to run:** run all cells top to bottom (Colab or local Jupyter), or `python inventory_agent.py`.
# If `data/sales.csv`, `data/inventory.csv`, `data/params.csv` are missing they are auto-generated (seeded).

# {{include:design_doc.md}}

# %% [markdown]
# ## 1. Setup & configuration
# Every tunable lives in `AgentConfig` (EWMA `alpha`, service level / lead time overrides, cover days, caps, order cost).

# %%
import os
import math
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

try:                                    # pretty tables in Jupyter, plain print in a terminal
    from IPython.display import display
except ImportError:                     # pragma: no cover
    display = print

DATA_DIR = "data"
OUT_DIR = "outputs"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


@dataclass
class AgentConfig:
    """All knobs for the agent and the simulation (change these, not the code)."""
    alpha: float = 0.1                  # EWMA smoothing for the demand level (higher = reacts faster); 0.1 was chosen on
                                        # tuning seeds 200-229, which are NOT used in the evaluation below
    beta: float = 0.1                   # EWMA smoothing for the squared forecast error (sigma estimate)
    forecaster: str = "ewma"            # "ewma" or "naive" (yesterday = today)
    review_period: int = 1              # days between reviews (the agent reviews daily)
    cover_days: int = 7                 # extra days of demand bought per PO (sets the order cycle)
    max_cover_days: int = 25            # GUARDRAIL: inventory position never exceeds this many days of demand + SS
    max_inventory_value: Optional[float] = None   # GUARDRAIL: optional cap on inventory position value ($) per SKU
    fixed_order_cost: float = 20.0      # cost per PO (assignment allows 0 or a constant)
    warmup_days: int = 14               # history used only to initialise the forecaster / baseline (not simulated)
    service_level_override: Optional[float] = None   # if set, replaces service_level in params.csv for every SKU
    lead_time_override: Optional[int] = None         # if set, replaces lead_time_days in params.csv for every SKU


# %% [markdown]
# ## 2. Data: generate (if missing), load, aggregate
# Three CSVs with the schemas required by the assignment. `make_synthetic` builds 3 SKUs × 120 days with weekly
# seasonality, a trend, promo spikes and noise. A few days are split across two sales rows on purpose, so the
# aggregation step (sum per day per SKU) does real work.

# %%
SKU_PROFILES = {
    #  base = mean units/day, trend = relative change over the horizon, promos = (count, lift)
    "SKU_A": dict(base=40, trend=0.35, promos=(3, 0.9), opening=300,
                  params=dict(unit_cost=12.0, holding_cost_per_day=0.05, stockout_cost=9.0,
                              lead_time_days=5, min_order_qty=60, service_level=0.95)),
    "SKU_B": dict(base=18, trend=0.00, promos=(3, 1.2), opening=150,
                  params=dict(unit_cost=35.0, holding_cost_per_day=0.14, stockout_cost=28.0,
                              lead_time_days=7, min_order_qty=40, service_level=0.97)),
    "SKU_C": dict(base=6, trend=-0.20, promos=(2, 1.5), opening=40,
                  params=dict(unit_cost=8.0, holding_cost_per_day=0.03, stockout_cost=4.0,
                              lead_time_days=3, min_order_qty=20, service_level=0.90)),
}
DOW_FACTOR = np.array([1.00, 0.95, 1.00, 1.05, 1.20, 1.25, 0.80])   # Mon..Sun


def make_synthetic(seed: int = 7, n_days: int = 120, start: str = "2025-01-01"):
    """Return (sales, inventory, params) DataFrames in the assignment's CSV schemas."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_days, freq="D")
    rows, inv_rows, par_rows = [], [], []
    for sku, prof in SKU_PROFILES.items():
        t = np.arange(n_days) / (n_days - 1)
        mean = prof["base"] * DOW_FACTOR[dates.dayofweek] * (1 + prof["trend"] * t)
        n_promo, lift = prof["promos"]
        for d in rng.choice(np.arange(20, n_days - 2), size=n_promo, replace=False):
            mean[d:d + 2] *= (1 + lift)                                   # 2-day promo spike
        demand = rng.poisson(mean * rng.lognormal(0, 0.10, n_days))        # noisy integer demand
        for date, q in zip(dates, demand):
            if q >= 2 and rng.random() < 0.10:                             # split some days across 2 sales rows
                a = int(rng.integers(1, q))
                rows += [(date, sku, a), (date, sku, int(q) - a)]
            else:
                rows.append((date, sku, int(q)))
        inv_rows.append((sku, prof["opening"]))
        par_rows.append((sku, *prof["params"].values()))
    sales = pd.DataFrame(rows, columns=["date", "sku", "qty_sold"])
    inventory = pd.DataFrame(inv_rows, columns=["sku", "opening_stock"])
    params = pd.DataFrame(par_rows, columns=["sku", "unit_cost", "holding_cost_per_day", "stockout_cost",
                                             "lead_time_days", "min_order_qty", "service_level"])
    return sales, inventory, params


def write_data(data_dir: str = DATA_DIR, seed: int = 7):
    sales, inventory, params = make_synthetic(seed)
    sales.to_csv(f"{data_dir}/sales.csv", index=False)
    inventory.to_csv(f"{data_dir}/inventory.csv", index=False)
    params.to_csv(f"{data_dir}/params.csv", index=False)


def prepare_data(sales: pd.DataFrame, inventory: pd.DataFrame, params: pd.DataFrame):
    """Validate the inputs and aggregate raw sales rows into daily demand per SKU.

    Returns (daily, opening, params):  daily = DataFrame[date x sku] of units sold,
    opening = Series sku -> opening stock, params = DataFrame indexed by sku.
    """
    needed = {"sales": (sales, {"date", "sku", "qty_sold"}),
              "inventory": (inventory, {"sku", "opening_stock"}),
              "params": (params, {"sku", "unit_cost", "holding_cost_per_day", "stockout_cost",
                                  "lead_time_days", "min_order_qty", "service_level"})}
    for name, (df, cols) in needed.items():
        missing = cols - set(df.columns)
        if missing:
            raise ValueError(f"{name}.csv is missing columns: {sorted(missing)}")
    sales = sales.copy()
    sales["date"] = pd.to_datetime(sales["date"])
    if (sales["qty_sold"] < 0).any():
        raise ValueError("sales.csv contains negative qty_sold")
    skus = sorted(sales["sku"].unique())
    for name, df in (("inventory", inventory), ("params", params)):
        absent = set(skus) - set(df["sku"])
        if absent:
            raise ValueError(f"{name}.csv has no row for SKU(s): {sorted(absent)}")

    # Task 1: aggregate to one number per (day, SKU), then fill the calendar so there are no date gaps.
    daily = sales.groupby(["date", "sku"])["qty_sold"].sum().unstack("sku")[skus]
    full_calendar = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    gap_days = full_calendar.difference(daily.index)
    if len(gap_days):
        warnings.warn(f"{len(gap_days)} calendar day(s) have no sales rows; treating them as 0 demand.")
    daily = daily.reindex(full_calendar).fillna(0).astype(int)
    daily.index.name = "date"
    return daily, inventory.set_index("sku")["opening_stock"], params.set_index("sku")


def load_data(data_dir: str = DATA_DIR):
    """Read the three CSVs (auto-generating them if absent) and return prepared data."""
    files = [f"{data_dir}/{n}.csv" for n in ("sales", "inventory", "params")]
    if not all(os.path.exists(f) for f in files):
        print("CSV files not found - generating seeded synthetic data.")
        write_data(data_dir)
    return prepare_data(pd.read_csv(files[0]), pd.read_csv(files[1]), pd.read_csv(files[2]))


daily, opening, params = load_data()
print(f"Loaded {len(daily)} days x {daily.shape[1]} SKUs  ({daily.index.min().date()} -> {daily.index.max().date()})")
display(params)
display(opening.to_frame())
display(daily.head())

# %% [markdown]
# ## 3. Forecaster (EWMA or naive) and forecast error
# * **EWMA level:** `f_t+1 = alpha * d_t + (1 - alpha) * f_t` — updated every day after the day's demand is observed.
# * **Naive:** `f_t+1 = d_t` (yesterday = today).
# * **Forecast error:** `e_t = d_t - f_t` (the forecast that was held *before* seeing day *t*).
#   Its squared value is smoothed with its own EWMA (`beta`) to give a running error std-dev `sigma`,
#   which feeds the safety-stock formula.

# %%
class DemandForecaster:
    """Per-SKU one-step-ahead demand forecaster with a running forecast-error estimate."""

    def __init__(self, history, kind="ewma", alpha=0.3, beta=0.1):
        if kind not in ("ewma", "naive"):
            raise ValueError("kind must be 'ewma' or 'naive'")
        self.kind, self.alpha, self.beta = kind, alpha, beta
        history = np.asarray(history, dtype=float)
        self.level = history[0]
        sq_errors = []
        for d in history[1:]:                       # warm-up: replay history to initialise level and error variance
            sq_errors.append((d - self.level) ** 2)
            self._update_level(d)
        self.var = float(np.mean(sq_errors)) if sq_errors else float(np.var(history))
        self.last_error = 0.0

    def _update_level(self, actual):
        self.level = actual if self.kind == "naive" else self.alpha * actual + (1 - self.alpha) * self.level

    @property
    def forecast(self) -> float:
        """Forecast of average daily demand for the coming days."""
        return max(self.level, 0.0)

    @property
    def sigma(self) -> float:
        """Std-dev of one-day forecast error."""
        return math.sqrt(max(self.var, 1e-9))

    def update(self, actual):
        """Called once per day after demand is observed: update error estimate, then the forecast."""
        self.last_error = actual - self.level
        self.var = (1 - self.beta) * self.var + self.beta * self.last_error ** 2
        self._update_level(actual)


# %% [markdown]
# ## 4. The agent (decision logic) and the static baseline
# **Agent, every day for every SKU** (protection interval `H = lead_time + review_period` days):
# 1. *Observe* – on-hand after today's receipts, units on order (all arrive within `H`), forecast `f`, error `sigma`.
# 2. *Safety stock* – `SS = z(service_level) * sigma * sqrt(H)`.
# 3. *Project* – `projected = on_hand + on_order - f * H`  (inventory just before the next PO could land).
# 4. *Decide* – if `projected >= SS` → **WAIT** (ordering now only adds holding cost).
#    Otherwise → **ORDER**: bring the inventory position up to `f*H + SS + cover_days*f`, but
#    * never below `min_order_qty`, and
#    * never beyond the cap (`max_cover_days * f + SS`, and optionally `max_inventory_value`) – if the cap leaves no room for a
#      legal PO the agent waits and says so.
#
# **Baseline (static `(s, Q)`):** same service level, but mean/std are computed *once* from the warm-up window and never
# updated (no forecast). Order a fixed `Q = max(MOQ, cover_days * mean)` whenever inventory position `<= s`.

# %%
@dataclass
class SkuParams:
    """Policy and cost inputs for one SKU (after applying any config overrides)."""
    sku: str
    unit_cost: float
    h: float                 # holding cost per unit per day
    p: float                 # stockout cost per unit of lost demand
    L: int                   # lead time (days)
    moq: int                 # minimum order quantity
    service_level: float

    @classmethod
    def from_row(cls, sku, row, cfg: AgentConfig):
        return cls(sku=sku, unit_cost=float(row["unit_cost"]), h=float(row["holding_cost_per_day"]),
                   p=float(row["stockout_cost"]),
                   L=int(cfg.lead_time_override if cfg.lead_time_override is not None else row["lead_time_days"]),
                   moq=int(row["min_order_qty"]),
                   service_level=float(cfg.service_level_override if cfg.service_level_override is not None
                                       else row["service_level"]))


@dataclass
class Decision:
    action: str              # "ORDER" or "WAIT"
    qty: int
    forecast: float
    sigma: float
    safety_stock: float
    projected: float
    rationale: str


class ReplenishmentAgent:
    """Forecast-driven (s, S)-style agent with a dynamic reorder trigger and order-up-to level."""
    name = "EWMA agent"

    def __init__(self, sp: SkuParams, history, cfg: AgentConfig):
        self.sp, self.cfg = sp, cfg
        self.fc = DemandForecaster(history, cfg.forecaster, cfg.alpha, cfg.beta)
        self.z = norm.ppf(sp.service_level)

    def decide(self, on_hand: float, on_order: float) -> Decision:
        sp, cfg = self.sp, self.cfg
        f, sigma = self.fc.forecast, self.fc.sigma
        H = sp.L + cfg.review_period                                  # protection interval (days)
        demand_H = f * H
        safety = self.z * sigma * math.sqrt(H)                        # safety stock from forecast error + service level
        projected = on_hand + on_order - demand_H                     # projected inventory over lead time + review period
        head = (f"fc={f:.1f}/d sigma={sigma:.1f} SS={safety:.1f}; projected inv after {H}d = "
                f"{on_hand:.0f}+{on_order:.0f}-{demand_H:.1f} = {projected:.1f}")

        if projected >= safety:
            return Decision("WAIT", 0, f, sigma, safety, projected,
                            f"{head} >= SS -> enough cover, wait (saves holding cost)")

        position = on_hand + on_order
        target = demand_H + safety + cfg.cover_days * f               # order-up-to level
        cap = cfg.max_cover_days * f + safety                         # guardrail: no unrealistic over-ordering
        if cfg.max_inventory_value is not None:
            cap = min(cap, cfg.max_inventory_value / sp.unit_cost)
        room = cap - position
        if room < sp.moq:
            return Decision("WAIT", 0, f, sigma, safety, projected,
                            f"{head} < SS but cap {cap:.0f} leaves room {room:.0f} < MOQ {sp.moq} -> wait (over-order guardrail)")
        need = target - position
        qty = int(math.ceil(min(max(need, sp.moq), room)))
        notes = []
        if need < sp.moq:
            notes.append(f"raised to MOQ {sp.moq}")
        elif need > room:
            notes.append(f"trimmed to cap {cap:.0f}")
        return Decision("ORDER", qty, f, sigma, safety, projected,
                        f"{head} < SS -> ORDER {qty} (target {target:.0f}, position {position:.0f}"
                        f"{'; ' + ', '.join(notes) if notes else ''}); arrives in {sp.L}d")

    def observe(self, demand: float):
        self.fc.update(demand)                                        # forecast updates daily


class StaticReorderPolicy:
    """Baseline: fixed (s, Q) with s and Q computed once from the warm-up window. No forecasting."""
    name = "Static (s,Q) baseline"

    def __init__(self, sp: SkuParams, history, cfg: AgentConfig):
        self.sp = sp
        history = np.asarray(history, dtype=float)
        mu0, sd0 = history.mean(), history.std(ddof=1)
        H = sp.L + cfg.review_period
        self.s = mu0 * H + norm.ppf(sp.service_level) * sd0 * math.sqrt(H)
        self.Q = int(math.ceil(max(sp.moq, cfg.cover_days * mu0)))
        self.mu0, self.sd0 = mu0, sd0

    def decide(self, on_hand: float, on_order: float) -> Decision:
        position = on_hand + on_order
        if position <= self.s:
            return Decision("ORDER", self.Q, self.mu0, self.sd0, 0.0, position,
                            f"position {position:.0f} <= fixed s={self.s:.0f} -> ORDER fixed Q={self.Q}")
        return Decision("WAIT", 0, self.mu0, self.sd0, 0.0, position,
                        f"position {position:.0f} > fixed s={self.s:.0f} -> wait")

    def observe(self, demand: float):
        pass                                                          # static: never learns


def NaiveForecastAgent(sp, history, cfg):
    """Same agent logic, but with the 'yesterday = today' forecaster."""
    cfg = AgentConfig(**{**cfg.__dict__, "forecaster": "naive"})
    agent = ReplenishmentAgent(sp, history, cfg)
    agent.name = "Naive-forecast agent"
    return agent


# %% [markdown]
# ## 5. Day-by-day simulation
# Order of events each day *t* (per SKU):
# 1. **Receive** POs whose arrival day is *t* (an order placed on day *d* arrives on day *d + lead_time*).
# 2. **Decide** – the agent sees only data through day *t-1* and today's stock position; it may place a PO.
# 3. **Demand** occurs; units are sold from on-hand. Unmet demand is **lost** (no backorders) and costs `stockout_cost` per unit.
# 4. **Costs** – holding cost on end-of-day stock, stockout cost on lost units, fixed cost per PO.
# 5. **Learn** – the actual demand updates the forecaster.
#
# The first `warmup_days` only initialise the forecaster; the simulation runs on the remaining days.
# SKUs are independent (no shared budget), so each is simulated on its own and the results are merged.

# %%
def simulate(daily, opening, params, cfg: AgentConfig, policy_cls=ReplenishmentAgent) -> pd.DataFrame:
    """Simulate one policy over all SKUs; return the daily log (one row per date x SKU)."""
    records = []
    n = len(daily)
    for sku in daily.columns:
        sp = SkuParams.from_row(sku, params.loc[sku], cfg)
        demand = daily[sku].to_numpy()
        policy = policy_cls(sp, demand[:cfg.warmup_days], cfg)
        on_hand = float(opening[sku])
        pipeline = []                                                 # list of (arrival_day_index, qty)
        for t in range(cfg.warmup_days, n):
            arrivals = sum(q for a, q in pipeline if a == t)          # 1. receive
            pipeline = [(a, q) for a, q in pipeline if a > t]
            on_hand += arrivals
            on_order = sum(q for _, q in pipeline)

            dec = policy.decide(on_hand, on_order)                    # 2. decide
            order_cost = 0.0
            if dec.qty > 0:
                pipeline.append((t + sp.L, dec.qty))
                order_cost = cfg.fixed_order_cost

            d = int(demand[t])                                        # 3. demand
            sold = min(on_hand, d)
            lost = d - sold
            on_hand_start, on_hand = on_hand, on_hand - sold

            records.append(dict(                                      # 4. costs + log
                date=daily.index[t], sku=sku, forecast=dec.forecast, sigma=dec.sigma,
                safety_stock=dec.safety_stock, on_hand_start=on_hand_start, on_order=on_order,
                projected=dec.projected, action=dec.action, order_qty=dec.qty, arrivals=arrivals,
                demand=d, sold=sold, lost=lost, on_hand_end=on_hand, unit_cost=sp.unit_cost,
                holding_cost=sp.h * on_hand, stockout_cost=sp.p * lost, order_cost=order_cost,
                rationale=dec.rationale))
            policy.observe(d)                                         # 5. learn
    return pd.DataFrame(records).sort_values(["date", "sku"]).reset_index(drop=True)


def print_daily_log(log: pd.DataFrame, max_rows: Optional[int] = None):
    """Print the agent's day-by-day decisions with their rationale (transparency)."""
    rows = log if max_rows is None else log.head(max_rows)
    for r in rows.itertuples():
        print(f"{r.date:%Y-%m-%d} {r.sku} | oh={r.on_hand_start:5.0f} (+{r.arrivals:3.0f} in) "
              f"| dem={r.demand:3.0f} lost={r.lost:3.0f} | {r.action:5s} {r.order_qty:4.0f} | {r.rationale}")


# %% [markdown]
# ## 6. Metrics
# * **Stockout** – a day where demand exceeded stock (`stockout_days`) and the units lost (`units_short`).
# * **Fill rate** – `1 - lost units / demanded units` (share of demand served from stock).
# * **Total cost** – `holding + stockout + order` cost. Purchase cost is not included (every policy buys what it sells);
#   `unit_cost` is used for the *inventory value* (working capital) metrics instead.

# %%
def compute_kpis(log: pd.DataFrame) -> pd.DataFrame:
    """KPIs per SKU plus a TOTAL row, computed from the daily log."""
    g = log.assign(inv_value=log["on_hand_end"] * log["unit_cost"]).groupby("sku")
    k = pd.DataFrame({
        "demand": g["demand"].sum(), "units_short": g["lost"].sum(),
        "stockout_days": g["lost"].apply(lambda s: int((s > 0).sum())),
        "orders": g["order_qty"].apply(lambda s: int((s > 0).sum())),
        "holding_cost": g["holding_cost"].sum(), "stockout_cost": g["stockout_cost"].sum(),
        "order_cost": g["order_cost"].sum(),
        "avg_inv_value": g["inv_value"].mean(), "peak_inv_value": g["inv_value"].max()})
    total = k.sum()
    total["peak_inv_value"] = log.assign(v=log["on_hand_end"] * log["unit_cost"]).groupby("date")["v"].sum().max()
    k.loc["TOTAL"] = total
    k["fill_rate"] = 1 - k["units_short"] / k["demand"]
    k["total_cost"] = k["holding_cost"] + k["stockout_cost"] + k["order_cost"]
    return k[["demand", "units_short", "fill_rate", "stockout_days", "orders", "holding_cost", "stockout_cost",
              "order_cost", "total_cost", "avg_inv_value", "peak_inv_value"]]


def run_policies(daily, opening, params, cfg):
    """Simulate baseline, naive-forecast agent and EWMA agent; return {name: (log, kpis)}."""
    out = {}
    for cls in (StaticReorderPolicy, NaiveForecastAgent, ReplenishmentAgent):
        log = simulate(daily, opening, params, cfg, cls)
        out[cls.name] = (log, compute_kpis(log))
    return out


# NaiveForecastAgent is a factory function; give it a name so run_policies can label it.
NaiveForecastAgent.name = "Naive-forecast agent"

cfg = AgentConfig()                                   # defaults: alpha=0.1, cover_days=7, order cost=$20/PO
results = run_policies(daily, opening, params, cfg)
BASE, NAIVE, AGENT = "Static (s,Q) baseline", "Naive-forecast agent", "EWMA agent"

# %% [markdown]
# ## 7. Daily agent log (EWMA agent)
# One line per day per SKU: stock, demand, action, and the numbers behind the decision. The full log is also saved to
# `outputs/agent_daily_log.csv`.

# %%
agent_log = results[AGENT][0]
agent_log.to_csv(f"{OUT_DIR}/agent_daily_log.csv", index=False)
print_daily_log(agent_log)

# %% [markdown]
# ## 8. Results: agent vs baseline

# %%
summary = pd.DataFrame({name: kp.loc["TOTAL"] for name, (_, kp) in results.items()}).T
display(summary.round(3))
for name, (_, kp) in results.items():
    print(f"\n--- {name}: per SKU ---")
    display(kp.round(3))

b, a = summary.loc[BASE], summary.loc[AGENT]
print(f"\nEWMA agent vs static baseline:  total cost {a.total_cost:,.0f} vs {b.total_cost:,.0f} "
      f"({(a.total_cost / b.total_cost - 1):+.1%}),  fill rate {a.fill_rate:.1%} vs {b.fill_rate:.1%} "
      f"({(a.fill_rate - b.fill_rate) * 100:+.1f} pts),  units short {a.units_short:.0f} vs {b.units_short:.0f},  "
      f"avg inventory value ${a.avg_inv_value:,.0f} vs ${b.avg_inv_value:,.0f}")

# Charts -------------------------------------------------------------------------------------------------
COLORS = {BASE: "#7f7f7f", NAIVE: "#e69f00", AGENT: "#0072b2"}          # colour-blind-safe palette
fig, axes = plt.subplots(len(daily.columns), 1, figsize=(11, 8), sharex=True)
for ax, sku in zip(np.atleast_1d(axes), daily.columns):
    for name in (BASE, AGENT):
        lg = results[name][0].query("sku == @sku")
        ax.plot(lg["date"], lg["on_hand_end"], label=name, color=COLORS[name], lw=1.6)
    po = results[AGENT][0].query("sku == @sku and order_qty > 0")
    ax.scatter(po["date"], po["on_hand_end"], marker="v", color=COLORS[AGENT], s=28, zorder=3, label="agent PO placed")
    ax.set_ylabel(f"{sku}\non-hand (units)")
    ax.grid(alpha=0.25)
np.atleast_1d(axes)[0].legend(ncol=3, loc="upper right", fontsize=8)
np.atleast_1d(axes)[0].set_title("End-of-day inventory: static baseline vs EWMA agent")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/inventory_trajectories.png", dpi=130); plt.show()

fig, ax = plt.subplots(figsize=(8, 3.6))
bottom = np.zeros(len(results))
for col, label, color in (("holding_cost", "Holding", "#56b4e9"), ("stockout_cost", "Stockout", "#d55e00"),
                          ("order_cost", "Order", "#009e73")):
    vals = summary[col].to_numpy(dtype=float)
    ax.barh(summary.index, vals, left=bottom, label=label, color=color)
    bottom += vals
ax.set_xlabel("Total cost ($)"); ax.legend(); ax.set_title("Cost breakdown by policy")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/cost_breakdown.png", dpi=130); plt.show()

# %% [markdown]
# ## 9. Robustness and a fairness check
# Two honest questions about the headline result:
# 1. **Is it just luck of one dataset?** Re-generate demand with 30 other seeds (100-129; the default `alpha` was chosen on
#    different seeds, 200-229) and compare the policies on average.
# 2. **Is the baseline just under-stocked?** The static baseline is built from the same service level, but from a static
#    14-day estimate, so it can miss its own target when demand trends or spikes. We therefore also give the baseline a
#    *hand-tuned, much higher* service level (0.999) - what an analyst would do after seeing it stock out.

# %%
BASE_TUNED = "Static (s,Q), hand-tuned to SL=0.999"


def run_tuned_baseline(d, o, p, cfg, service_level=0.999):
    c = AgentConfig(**{**cfg.__dict__, "service_level_override": service_level})
    return compute_kpis(simulate(d, o, p, c, StaticReorderPolicy))


def multi_seed(seeds, cfg):
    rows = []
    for s in seeds:
        d, o, p = prepare_data(*make_synthetic(seed=s))
        kpis = {name: kp for name, (_, kp) in run_policies(d, o, p, cfg).items()}
        kpis[BASE_TUNED] = run_tuned_baseline(d, o, p, cfg)
        for name, kp in kpis.items():
            t = kp.loc["TOTAL"]
            rows.append(dict(seed=s, policy=name, total_cost=t.total_cost, fill_rate=t.fill_rate,
                             units_short=t.units_short, avg_inv_value=t.avg_inv_value))
    return pd.DataFrame(rows)


ms = multi_seed(range(100, 130), cfg)
ms_summary = ms.groupby("policy")[["total_cost", "fill_rate", "units_short", "avg_inv_value"]].agg(["mean", "std"]).round(3)
display(ms_summary)
piv = ms.pivot(index="seed", columns="policy", values="total_cost")
for ref in (BASE, BASE_TUNED):
    print(f"EWMA agent vs {ref:40s}: lower cost in {(piv[AGENT] < piv[ref]).mean():4.0%} of 30 seeds, "
          f"mean cost change {(piv[AGENT].mean() / piv[ref].mean() - 1):+.1%}")

# %% [markdown]
# ## 10. Sensitivity: the inventory-vs-service trade-off
# Sweep the target service level for **both** the agent and the static baseline (and the EWMA `alpha`) to see how much
# extra inventory each point of fill rate costs, and whether the agent sits on a better frontier than the baseline.

# %%
def service_sweep(policy_cls, levels=(0.80, 0.90, 0.95, 0.97, 0.99, 0.999)):
    rows = []
    for sl in levels:
        kp = compute_kpis(simulate(daily, opening, params, AgentConfig(service_level_override=sl), policy_cls)).loc["TOTAL"]
        rows.append(dict(service_level=sl, fill_rate=kp.fill_rate, units_short=kp.units_short,
                         holding_cost=kp.holding_cost, stockout_cost=kp.stockout_cost,
                         total_cost=kp.total_cost, avg_inv_value=kp.avg_inv_value))
    return pd.DataFrame(rows)


sweep = service_sweep(ReplenishmentAgent)
sweep_base = service_sweep(StaticReorderPolicy)
print("EWMA agent - target service level sweep"); display(sweep.round(3))
print("Static baseline - target service level sweep"); display(sweep_base.round(3))

alpha_sweep = []
for al in (0.05, 0.1, 0.2, 0.3, 0.5, 0.8):
    kp = compute_kpis(simulate(daily, opening, params, AgentConfig(alpha=al))).loc["TOTAL"]
    alpha_sweep.append(dict(alpha=al, fill_rate=kp.fill_rate, total_cost=kp.total_cost, avg_inv_value=kp.avg_inv_value))
print("EWMA agent - alpha sweep (this dataset; one run, so noisy)"); display(pd.DataFrame(alpha_sweep).round(3))

fig, ax = plt.subplots(figsize=(7.5, 4))
for df_, name in ((sweep, AGENT), (sweep_base, BASE)):
    ax.plot(df_["avg_inv_value"], df_["fill_rate"] * 100, "o-", color=COLORS[name], label=f"{name} (service level sweep)")
    for r in df_.itertuples():
        ax.annotate(f"{r.service_level:g}", (r.avg_inv_value, r.fill_rate * 100), textcoords="offset points",
                    xytext=(4, -11), fontsize=7, color=COLORS[name])
ax.set_xlabel("Average inventory value ($)"); ax.set_ylabel("Fill rate (%)"); ax.grid(alpha=0.25); ax.legend(fontsize=8)
ax.set_title("Service vs inventory: labels are the target service level")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/tradeoff_curve.png", dpi=130); plt.show()

# {{include:tradeoffs.md}}

# {{include:scaling_note.md}}
