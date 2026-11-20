"""GTA VI trophy-rarity forecaster (make.fun gta6-trophy-rarity-forecast-01).

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, five
quantiles of its Exophase earn rate (percent of tracked players holding it)
at the case's target, from the house's recorded reads of the Exophase page
in the case's history. Python 3.13, standard library only, no network,
deterministic (no randomness, fixed file order, pure float arithmetic).

METHOD (see README.md for the full write-up, assumptions and sources)
----------------------------------------------------------------------
An earn rate is a ratio: rate(t) = E(t) / N(t), earners over tracked
players. Both move, and they move in opposite directions for hard trophies:
new players syncing profiles (N grows fast after a record launch) dilute
every rate, while the launch cohort's playtime grows E. So the model fits
the two sides separately and recombines them, instead of extrapolating the
ratio alone:

1.  Dedupe the history into distinct states (Exophase refreshes hours
    apart; most 10-minute reads repeat the previous page byte for byte).
    Time is x = ln(1 + hours since release, 2026-11-19T00:00Z).
2.  Fit log N(t) on x by least squares (slope damped x0.85: tracked-player
    growth decelerates after the launch surge) and extrapolate N(target).
3.  Per trophy, fit TWO trends on the distinct states, both in centroid
    form, recency-weighted (quadratic ramp 0.25 -> 1.0: the recent
    regime is what the target window looks like), with each slope shrunk 30% toward
    its trophy-class mean (hierarchical shrinkage: story chapters,
    story missions, stranger strands, grinds, activities, combat,
    money, misc):
      A. logit(rate) on x  (the house curve baseline's shape, but fitted
         on all history, not just the last 24 h);
      B. log(E) on x, E = rate * N / 100, recombined as
         rate = 100 * E(target) / N(target)  (dilution-aware).
    The median is the mean of A and B in logit space.
4.  Blend the data fit with a prior curve. The data weight grows with    the number of distinct states but shrinks when the horizon is long
    relative to the history span: two days of reads cannot contain what
    the launch cohort does on days 3-7 (a 50-80 h story means chapter
    thresholds are crossed after the checkpoint), so the prior carries
    real weight on the final's 4.5-day horizon. The prior is a
    logit-linear interpolation in ln(1+t) between two anchors per
    trophy: the live Exophase rates of 2026-11-20 ~05:37 UTC (about
    30 h after release, read for this entry) and a week-one (t = 168 h)
    level set from GTA V analogues (Steam: first mission 81.9%, final
    mission 23.8% lifetime, 100% completion 6.0%; PS4 final mission
    16.8%) scaled down for a 7-day window and a 50-80 h main story
    (reviewers averaged 73 h, platinum ~200 h). The prior is what
    carries flat-history cases (the practice cases, whose reads never
    change): there the median is the prior, floored so it cannot fall
    below last_rate * 0.5 / D (D = (1 + horizon_hours / 1.5) ** 0.85,
    the expected dilution) -- earners cannot leave, so a rate can only
    fall as fast as N grows.
5.  Hard constraints on the medians: the story chain is monotone
    (Chapter 1 >= Chapter 2 >= Chapter 3 >= Chapter 4 >= final mission,
    isotonic pool-adjacent-violators); platinum <= 100%-completion gold
    <= final mission; the intro trophy (Gone Fishing, ~100% from the
    first read) stays >= 99%; every median is capped at a per-class
    week-one ceiling.
6.  Spread: per-trophy logit sigma from the fit residuals, the N-fit
    residual and a random-walk term 0.28 * sqrt(horizon_days), floored
    at 0.45 and capped at 1.15 (flat history: 1.0-1.1, a refresh shock
    is the dominant uncertainty there). Quantiles are
    inv_logit(logit(median) + z_tau * sigma), rounded to 4 decimals.

The house baselines this must beat: persistence (median = last rate,
spread 0.5 * sqrt(days)) and a last-24h logit-linear curve fit. This
entry beats them by (a) fitting earners and tracked players separately,
(b) using all history with shrinkage instead of 24 h of noisy per-trophy
fits, (c) enforcing the story/platinum ordering, and (d) a tighter,
residual-based spread instead of a fixed 0.5/sqrt-day random walk.

Output: /output/<id>.out per case, {"trophies": [{"id": n,
"q": [q05, q25, q50, q75, q95]}, ...]}, at most 65,536 bytes.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a
# sibling module would import; this entry is one file, but keep the line
# harmless and identical in spirit to the starter.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
# Live Exophase snapshot read for this entry (2026-11-20 ~05:37 UTC),
# t ~ 29.6 h after release. Only the rates are used, as prior anchors.
ANCHOR_T_HOURS = 29.6
TARGET_WEEK_HOURS = 168.0


def clamp(x: float, lo: float = 0.01, hi: float = 99.99) -> float:
    return min(hi, max(lo, x))


def logit(pct: float) -> float:
    p = clamp(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float) -> float | int:
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(s: str) -> datetime:
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def ols(xs: list[float], ys: list[float],
        ws: list[float] | None = None) -> tuple[float, float, float]:
    """Weighted least squares y ~ a + b*x; returns (ybar, b, residual std).

    Centroid form is used by callers: ybar + b*(x - xbar). Weights default
    to 1; callers pass a recency ramp so the recent regime (dilution
    slowing, earners accumulating) counts more than launch night.
    """
    n = len(xs)
    if ws is None:
        ws = [1.0] * n
    wsum = sum(ws)
    xbar = sum(w * x for w, x in zip(ws, xs)) / wsum
    ybar = sum(w * y for w, y in zip(ws, ys)) / wsum
    sxx = sum(w * (x - xbar) ** 2 for w, x in zip(ws, xs))
    if sxx <= 0:
        return ybar, 0.0, 0.0
    b = sum(w * (x - xbar) * (y - ybar) for w, x, y in zip(ws, xs, ys)) / sxx
    resid = [y - (ybar + b * (x - xbar)) for x, y in zip(xs, ys)]
    var = sum(w * r * r for w, r in zip(ws, resid)) / wsum
    if n > 2:
        var *= n / (n - 2)  # small-sample inflation
    return ybar, b, math.sqrt(var)


# ---------------------------------------------------------------- priors
# Per trophy: (live Exophase rate ~30 h after release, 2026-11-20 05:37 UTC;
# week-one prior at t = 168 h, 2026-11-26 00:00 UTC). Week-one levels are
# judgement from the GTA V analogues and the story length (see module
# docstring / README.md): a 50-80 h story means almost nobody finishes in
# week one, while the intro and the early fishing challenge are earned by
# nearly every new player within hours, so they track N instead of
# diluting. Trophy ids follow Exophase's list order in the case files.
PRIORS: dict[int, tuple[float, float]] = {
    1: (0.03, 0.05), 2: (99.99, 99.90), 3: (0.73, 2.50), 4: (0.14, 0.60),
    5: (0.06, 0.30), 6: (0.06, 0.25), 7: (0.06, 0.25), 8: (0.24, 0.80),
    9: (0.06, 0.25), 10: (0.06, 0.25), 11: (0.06, 0.25), 12: (0.06, 0.25),
    13: (0.06, 0.25), 14: (0.03, 0.06), 15: (0.03, 0.08), 16: (0.04, 0.30),
    17: (0.04, 0.10), 18: (0.03, 0.08), 19: (0.04, 0.20), 20: (1.89, 3.00),
    21: (0.41, 1.00), 22: (0.04, 0.15), 23: (0.04, 0.15), 24: (0.04, 0.15),
    25: (0.03, 0.08), 26: (0.03, 0.12), 27: (0.04, 0.15), 28: (0.03, 0.08),
    29: (0.04, 0.12), 30: (0.03, 0.06), 31: (0.03, 0.06), 32: (0.03, 0.08),
    33: (0.09, 0.40), 34: (0.05, 0.25), 35: (0.24, 0.90), 36: (0.06, 0.30),
    37: (0.14, 0.50), 38: (0.14, 0.50), 39: (0.06, 0.25), 40: (0.04, 0.12),
    41: (0.04, 0.20), 42: (0.04, 0.20), 43: (0.04, 0.12), 44: (0.08, 0.40),
    45: (0.04, 0.20), 46: (0.09, 0.40), 47: (0.04, 0.15), 48: (0.05, 0.20),
    49: (0.04, 0.15), 50: (0.11, 0.45), 51: (0.06, 0.25), 52: (0.04, 0.12),
    53: (0.05, 0.25), 54: (0.04, 0.12), 55: (0.04, 0.20),
}

# Week-one ceilings (percent) per trophy class: a median above these in
# seven days would contradict the story length / GTA V analogues.
CEILINGS = {
    "platinum": 0.6, "tutorial": 99.99, "chapter": 25.0, "final": 5.0,
    "mission": 4.0, "stranger": 2.0, "grind": 2.0, "activity": 6.0,
    "combat": 4.0, "money": 3.0, "misc": 3.0, "completion": 1.0,
}

# Earner-growth multiplier used only in the flat-history dilution floor
# (see forecast_case): how much earners can grow over the horizon by class.
GROWTH = {
    "platinum": 1.05, "tutorial": 1.0, "chapter": 1.15, "final": 1.1,
    "mission": 1.15, "stranger": 1.1, "grind": 1.05, "activity": 1.3,
    "combat": 1.2, "money": 1.2, "misc": 1.2, "completion": 1.05,
}


def classify(trophy: dict) -> str:
    """Trophy class from tier + name + description (no id hard-coding)."""
    tier = str(trophy.get("tier", "")).lower()
    name = str(trophy.get("name", "")).lower()
    desc = str(trophy.get("description", "")).lower()
    if tier == "platinum":
        return "platinum"
    if "gone fishing" in desc:
        return "tutorial"
    if desc.startswith("complete chapter"):
        return "chapter"
    if desc.startswith("complete the final mission"):
        return "final"
    if "100% completion" in desc:
        return "completion"
    if "final mission for" in desc or "mission strands for strangers" in desc:
        return "stranger"
    if any(k in desc for k in ("250 stunts", "100 locations", "50 challenges",
                               "collectible card sets", "gold medal in all",
                               "every location", "20 different races")):
        return "grind"
    if any(k in desc for k in ("try each type", "arcade game",
                               "fishing challenge", "meditate at every spot",
                               "unique animal", "discover 100 vehicles",
                               "explore all of leonida")):
        return "activity"
    if any(k in desc for k in ("take down 5 enemies", "rhinos",
                               "jewelry store", "bank vaults", "armored truck",
                               "illegal operation", "wanted level",
                               "unscheduled departure", "melee drive-by")):
        return "combat"
    if any(k in desc for k in ("laundering", "spend $100,000",
                               "garage properties", "register 10 vehicles")):
        return "money"
    if desc.startswith("complete ") and trophy.get("secret"):
        return "mission"
    return "misc"


def prior_rate(trophy_id: int, t_hours: float) -> float:
    """Prior median (percent) at t_hours after release, from the anchors."""
    live, week1 = PRIORS.get(trophy_id, (0.1, 0.2))
    x1 = math.log1p(ANCHOR_T_HOURS)
    x2 = math.log1p(TARGET_WEEK_HOURS)
    y1, y2 = logit(live), logit(week1)
    slope = (y2 - y1) / (x2 - x1)
    x = math.log1p(max(0.0, t_hours))
    if x <= x1:
        y = y1 + slope * (x - x1)
    elif x <= x2:
        y = y1 + slope * (x - x1)
    else:  # past the week-one anchor: damp the slope, rates saturate
        y = y2 + 0.5 * slope * (x - x2)
    return clamp(inv_logit(y), 0.01, 99.9)


def isotonic_decreasing(values: list[float]) -> list[float]:
    """Pool-adjacent-violators: smallest decreasing sequence >= values in
    the least-squares sense (here: averages pooled blocks)."""
    blocks: list[list[float]] = [[v] for v in values]
    i = 0
    while i < len(blocks) - 1:
        mean_i = sum(blocks[i]) / len(blocks[i])
        mean_j = sum(blocks[i + 1]) / len(blocks[i + 1])
        if mean_i < mean_j:  # violates decreasing: pool them
            blocks[i] = blocks[i] + blocks[i + 1]
            del blocks[i + 1]
            if i > 0:
                i -= 1
        else:
            i += 1
    out: list[float] = []
    for b in blocks:
        m = sum(b) / len(b)
        out.extend([m] * len(b))
    return out


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case["history"]
    if not history:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    horizon_h = case["horizon_minutes"] / 60.0
    horizon_d = case["horizon_minutes"] / 1440.0
    target_dt = parse_time(case["target"])
    t_target = (target_dt - RELEASE).total_seconds() / 3600.0

    # --- distinct states: collapse consecutive identical reads ---------
    states: list[dict] = []
    for read in history:
        key = (read["players_tracked"], tuple(read["rates"]))
        if states and states[-1]["_key"] == key:
            continue
        row = dict(read)
        row["_key"] = key
        row["_t"] = (parse_time(read["at"]) - RELEASE).total_seconds() / 3600.0
        row["_x"] = math.log1p(max(0.0, row["_t"]))
        states.append(row)
    n_states = len(states)
    last = states[-1]
    n_last = last["players_tracked"]

    # --- fit log N(t) ---------------------------------------------------
    # Strong recency ramp (quadratic, 0.25 -> 1.0 over the distinct
    # states): the recent regime -- tracked growth decelerating after the
    # launch surge, rates bottoming out as earners catch up -- is what
    # the target window looks like, not launch night's backlog sync and
    # first-refresh crash. A flat average over all history would drag
    # every forecast down the early dilution slope.
    t_first, t_last = states[0]["_t"], states[-1]["_t"]
    span_rec = max(1e-9, t_last - t_first)
    recency = [0.25 + 0.75 * ((s["_t"] - t_first) / span_rec) ** 2 for s in states]
    state_w = {id(s): w for s, w in zip(states, recency)}
    xs_n = [s["_x"] for s in states if s["players_tracked"] > 0]
    ys_n = [math.log(s["players_tracked"]) for s in states if s["players_tracked"] > 0]
    ws_n = [state_w[id(s)] for s in states if s["players_tracked"] > 0]
    if len(set(ys_n)) >= 2:
        ybar_n, b_n, s_n = ols(xs_n, ys_n, ws_n)
        b_n_eff = b_n * 0.85  # growth decelerates after the launch surge
        x_t = math.log1p(max(0.0, t_target))
        xbar_n = sum(w * x for w, x in zip(ws_n, xs_n)) / sum(ws_n)
        log_n_target = ybar_n + b_n_eff * (x_t - xbar_n)
        n_target = math.exp(log_n_target)
        n_target = min(max(n_target, float(n_last)), n_last * 30.0)
    else:
        s_n, n_target = 0.0, float(n_last)

    # --- per-trophy fits ------------------------------------------------
    classes = [classify(t) for t in trophies]
    # First pass: raw per-trophy slopes, for the class-mean shrinkage.
    fits: list[dict] = []
    for i, trophy in enumerate(trophies):
        xs_a, ys_a, ws_a, xs_b, ys_b, ws_b = [], [], [], [], [], []
        for s in states:
            r = s["rates"][i]
            if r is None:
                continue
            xs_a.append(s["_x"])
            ys_a.append(logit(r))
            ws_a.append(state_w[id(s)])
            if s["players_tracked"] > 0 and r > 0:
                xs_b.append(s["_x"])
                ys_b.append(math.log(max(1e-9, r / 100.0 * s["players_tracked"])))
                ws_b.append(state_w[id(s)])
        fa = ols(xs_a, ys_a, ws_a) if len(xs_a) >= 2 and len(set(xs_a)) >= 2 else None
        fb = ols(xs_b, ys_b, ws_b) if len(xs_b) >= 2 and len(set(xs_b)) >= 2 else None
        fits.append({"fa": fa, "fb": fb, "xs_a": xs_a, "ys_a": ys_a,
                     "xs_b": xs_b, "ys_b": ys_b, "ws_a": ws_a, "ws_b": ws_b})
    # Class-mean slopes (weighted by number of points).
    class_slope_a: dict[str, float] = {}
    class_slope_b: dict[str, float] = {}
    for cls in sorted(set(classes)):
        num_a = den_a = num_b = den_b = 0.0
        for i, cls_i in enumerate(classes):
            if cls_i != cls:
                continue
            if fits[i]["fa"] is not None:
                w = len(fits[i]["xs_a"])
                num_a += w * fits[i]["fa"][1]
                den_a += w
            if fits[i]["fb"] is not None:
                w = len(fits[i]["xs_b"])
                num_b += w * fits[i]["fb"][1]
                den_b += w
        if den_a:
            class_slope_a[cls] = num_a / den_a
        if den_b:
            class_slope_b[cls] = num_b / den_b

    x_t = math.log1p(max(0.0, t_target))
    dilution = (1.0 + horizon_h / 1.5) ** 0.85
    flat = n_states < 2
    # Data weight: grows with the number of distinct states, but shrinks
    # when the horizon is long relative to the history span -- two days of
    # reads cannot contain what the launch cohort does on days 3-7 (the
    # 50-80 h story means chapter thresholds are crossed after the
    # checkpoint), so the GTA V / story-length prior carries real weight
    # there. Practice cases are flat, so w_data = 0 for them regardless.
    span_h = max(0.0, states[-1]["_t"] - states[0]["_t"])
    horizon_factor = 0.5 + 0.5 * math.sqrt(span_h / (span_h + horizon_h)) if span_h > 0 else 0.5
    w_data = (n_states / (n_states + 3.0)) * horizon_factor if not flat else 0.0

    medians: list[float] = []
    sigmas: list[float] = []
    for i, trophy in enumerate(trophies):
        tid = trophy["id"]
        cls = classes[i]
        last_rate = last["rates"][i]
        last_rate = 0.0 if last_rate is None else float(last_rate)
        prior = prior_rate(tid, t_target)

        if last_rate >= 99.5:
            # Intro trophy: pinned near 100% from the first read on. The
            # page prints 2 decimals, so 99.99 is the effective ceiling
            # (logit(100) is clamped to logit(99.99) by the scorer too);
            # a median of 99.9 would cost ~2.3 logit units against a
            # 99.99 answer, so pin the median at 99.99, tight spread.
            medians.append(99.99)
            sigmas.append(0.05)
            continue

        if flat:
            med = prior
            lo = last_rate * 0.5 / dilution
            hi = min(99.9, last_rate * 3.0)
            med = min(max(med, lo), hi)
            med = min(med, CEILINGS[cls])
            medians.append(max(0.01, med))
            sigmas.append(1.0 if horizon_h <= 12 else 1.1)
            continue

        # Method A: logit(rate) trend, slope shrunk toward class mean.
        logit_med_a = None
        resid_a = 0.15
        f = fits[i]
        if f["fa"] is not None:
            ybar, b, resid_a = f["fa"]
            b_shrunk = 0.7 * b + 0.3 * class_slope_a.get(cls, b)
            xbar = sum(w * x for w, x in zip(f["ws_a"], f["xs_a"])) / sum(f["ws_a"])
            logit_med_a = ybar + b_shrunk * (x_t - xbar)
        # Method B: log(earners) trend, recombined over N(target).
        logit_med_b = None
        if f["fb"] is not None:
            ybar, b, _resid_b = f["fb"]
            b_shrunk = 0.7 * b + 0.3 * class_slope_b.get(cls, b)
            xbar = sum(w * x for w, x in zip(f["ws_b"], f["xs_b"])) / sum(f["ws_b"])
            log_e_target = ybar + b_shrunk * (x_t - xbar)
            rate_b = 100.0 * math.exp(log_e_target) / max(1.0, n_target)
            logit_med_b = logit(rate_b)
        cands = [v for v in (logit_med_a, logit_med_b) if v is not None]
        if cands:
            logit_med = sum(cands) / len(cands)
        else:
            logit_med = logit(prior)
        # Blend with the prior; data dominates once history is rich.
        logit_med = w_data * logit_med + (1.0 - w_data) * logit(prior)
        med = clamp(inv_logit(logit_med), 0.01, 99.9)
        med = min(med, CEILINGS[cls])
        medians.append(max(0.01, med))

        rw = 0.28 * math.sqrt(horizon_d)
        sig = math.sqrt(rw ** 2 + resid_a ** 2 + 0.25 * s_n ** 2 + 0.15 ** 2)
        # Long horizons cross a regime change (dilution giving way to the
        # cohort crossing playtime thresholds), so the spread floor rises.
        sig_floor = 0.60 if horizon_h > 72 else 0.45
        sigmas.append(min(1.15, max(sig_floor, sig)))

    # --- ordering constraints on the medians ---------------------------
    by_id = {t["id"]: k for k, t in enumerate(trophies)}
    chain = [by_id[k] for k in (3, 4, 5, 6, 7) if k in by_id]
    if len(chain) == 5:
        fixed = isotonic_decreasing([medians[k] for k in chain])
        for k, v in zip(chain, fixed):
            medians[k] = v
    if 14 in by_id and 7 in by_id:
        medians[by_id[14]] = min(medians[by_id[14]], medians[by_id[7]])
    if 1 in by_id and 14 in by_id:
        medians[by_id[1]] = min(medians[by_id[1]], medians[by_id[14]])
    # The intro trophy stays on top.
    if 2 in by_id:
        top = medians[by_id[2]]
        for k in range(len(medians)):
            if k != by_id[2]:
                medians[k] = min(medians[k], top)

    rows = []
    for i, trophy in enumerate(trophies):
        lg = logit(medians[i])
        sig = sigmas[i]
        qs = [round4(clamp(inv_logit(lg + z * sig))) for z in Z]
        # Rounding must not break the non-decreasing order.
        for j in range(1, 5):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
