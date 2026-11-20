# GTA VI Trophy Rarity Forecast — make.fun entry

**Competition:** `gta6-trophy-rarity-forecast-01` — forecast, for every trophy of
Grand Theft Auto VI on PlayStation 5, its Exophase earn rate at
2026-11-26T00:00:00Z (a week after launch), as five quantiles per trophy,
scored by mean pinball loss on the logit scale. $250, 20 places.
**Deadline:** 2026-11-21T11:30:00Z. **Entrypoint:** `main.py` (Python 3.13,
standard library only, no network, deterministic).

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` in the input directory, writes `<id>.out` per case:
`{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`, one row per
trophy, quantiles non-decreasing, each file ≤ 65,536 bytes. Two runs on the
same case write byte-identical output (no randomness, fixed file order).

## Method (summary)

An earn rate is a ratio, `rate(t) = E(t) / N(t)` — earners over tracked
players — and the two sides move in opposite directions for hard trophies:
new players syncing profiles dilute every rate, while the launch cohort's
playtime grows `E`. The model therefore fits the two sides separately and
recombines them, instead of extrapolating the ratio alone:

1. **Dedupe** the history into distinct states (Exophase refreshes hours
   apart; most 10-minute reads repeat the previous page). Time is
   `x = ln(1 + hours since release, 2026-11-19T00:00Z)`.
2. **Fit `log N(t)`** on `x` by least squares (slope damped ×0.85: tracked
   growth decelerates after the launch surge) and extrapolate `N(target)`.
3. **Per trophy, two trends**, both in centroid form, recency-weighted
   (quadratic ramp 0.25 → 1.0: the recent regime is what the target window
   looks like), each slope shrunk 30% toward its trophy-class mean
   (story chapters, story missions, stranger strands, grinds, activities,
   combat, money, misc):
   - **A.** `logit(rate)` on `x` (the house curve baseline's shape, but
     fitted on all history, not just the last 24 h);
   - **B.** `log(E)` on `x`, recombined as `rate = 100 · E(target)/N(target)`
     (dilution-aware). The median is the mean of A and B in logit space.
4. **Blend with a prior curve.** The data weight grows with the number of
   distinct states but shrinks when the horizon is long relative to the
   history span: two days of reads cannot contain what the launch cohort
   does on days 3–7 (a 50–80 h story means chapter thresholds are crossed
   after the checkpoint), so the prior carries real weight on the final's
   4.5-day horizon. The prior is a logit-linear interpolation in
   `ln(1+t)` between two anchors per trophy (see Data below). It is what
   carries flat-history cases (the practice cases, whose reads never
   change): there the median is the prior, floored so it cannot fall
   below `last_rate · 0.5 / D` with `D = (1 + horizon_hours/1.5)^0.85` —
   earners cannot leave, so a rate can only fall as fast as `N` grows.
5. **Hard constraints on the medians:** the story chain is monotone
   (Chapter 1 ≥ Chapter 2 ≥ Chapter 3 ≥ Chapter 4 ≥ final mission,
   isotonic pool-adjacent-violators); platinum ≤ 100%-completion gold ≤
   final mission; the intro trophy (Gone Fishing, ~100% from the first
   read) is pinned at 99.99% (the page prints 2 decimals; a median of
   99.9 would cost ~2.3 logit units against a 99.99 answer); every median
   is capped at a per-class week-one ceiling.
6. **Spread:** per-trophy logit sigma from the fit residuals, the N-fit
   residual and a random-walk term `0.28 · sqrt(horizon_days)`, floored at
   0.45 (0.60 when the horizon exceeds 72 h — a long horizon crosses a
   regime change) and capped at 1.15; flat history uses 1.0–1.1 (a
   refresh shock dominates there). Quantiles are
   `inv_logit(logit(median) + z_tau · sigma)`, rounded to 4 decimals.

**Why this beats the house baselines.** Persistence says every rate stays
put with a fixed `0.5·sqrt(days)` spread; the curve baseline fits each
trophy's logit rate on the last 24 h only. This entry fits earners and
tracked players separately (so a refresh that multiplies `N` does not read
as trophies being un-earned), uses all history with shrinkage instead of
24 h of noisy per-trophy fits, enforces the story/platinum ordering, and
uses a tighter residual-based spread instead of a fixed random walk.

## Data (real, with sources)

The trophy list itself is **not** predicted: the case files name all 55
trophies (1 platinum, 2 gold, 6 silver, 46 bronze, 28 hidden), and the
model reads them from the case. What was researched:

| Fact | Value | Source |
|---|---|---|
| GTA V, Steam achievements (Nov 2026) | First mission "Welcome to Los Santos" 81.9%; final mission 23.8%; 100% "Career Criminal" 6.0% | Steam global achievement data, cited Nov 2026 (via social summaries of Steam data) |
| GTA V, PS4 | Final mission 16.8% | PS4 trophy data, cited Nov 2026 |
| GTA V, PocketPSN sample | First mission 38.6%; final 7.4%; 100% 0.2%; platinum 0.1% | pocketpsn.com GTA V page (small self-selected sample — treated as a lower-bound check, not the anchor) |
| GTA VI launch sales | ~24.6M copies day one, ~$2.36B, ~75% PlayStation (~18.4M PS5) | Sensor Tower estimate via Deadline/Fox 5, 19–20 Nov 2026 (third-party estimate; Take-Two published nothing) |
| GTA VI story length | Main story 50–65 h (reviewers avg 73 h incl. side content); Rob Nelson (Rockstar North) 80 h with side content; 100%/platinum ~200 h; 5 chapters, 90 missions | GameSpot, ScreenRant, Dexerto, Beebom, esports.gg, Nov 2026 |
| Exophase live snapshot, 2026-11-20 ~05:37 UTC (~30 h after release) | All 55 rates, e.g. Ch.1 0.73%, Ch.2 0.14%, Ch.3/Ch.4/final 0.06%, platinum 0.03%, 100% 0.03%, Gone Fishing 99.99%, Beat Cal (fishing) 1.89%, arcade 0.41% | https://www.exophase.com/game/grand-theft-auto-vi-psn/trophies/ (fetched 20 Nov 2026; players-tracked header was not captured in the fetch — noted as a gap, the model does not use it) |
| Launch-night recording (kit cases p00/p01) | 638 players tracked, rates unchanged 23:50–04:00 UTC; first refresh 04:30 UTC → 3,868 players, every rate but the 100% one fell | Kit `input/p00.in`, `p01.in`, `RULES.txt` |

**Honest limits.** No official trophy list was published by Rockstar before
launch; the 55 names/descriptions come from the house's Exophase recording
and agree across outlets (GTABase, PowerPyx, Insider Gaming, EGamers).
Exophase's players-tracked count at the live snapshot is unknown (see
above), so the prior's earner counts are not pinned to it — the prior is
used only as a curve shape and a blend weight, never as the forecast. The
week-one prior levels are judgement from the GTA V analogues scaled down
for a 7-day window and a 50–80 h story; they are documented per trophy in
`main.py` (`PRIORS`) and blended with — never substituted for — the case
history on the final.

## Validation (all local, reproducible)

- **Determinism:** ran twice on the kit practice input; outputs
  byte-identical (`diff -r` clean).
- **Shape:** 55 rows per case, exactly one `trophies` key, five
  non-decreasing quantiles in [0, 100] per row, files 2,789 bytes (cap
  65,536). Story chain monotone; platinum ≤ 100% ≤ final mission.
- **Practice behaviour (p00/p01, flat history):** medians drop from the
  launch-night rates, as a refresh must — e.g. Chapter 1: 1.25% → 0.35%
  median (q05–q95 0.07–1.77%); platinum 0.63% → 0.17% (floor: earners
  cannot leave, so at N=3,868 the rate cannot be below 0.10%); Beat Cal
  2.04% → 1.18%; Gone Fishing stays 99.99%. The house persistence
  baseline scores ~1.51 loss on the practice because its narrow band
  misses the refresh entirely; this entry's median sits near the
  dilution-implied level with a 1.0-logit spread.
- **Backtest vs the live 30 h snapshot** (flat history, target Nov 20
  05:37 UTC): median logit error 0.00 — **but this is circular at the
  anchor by construction** (the prior's first anchor IS that snapshot);
  it only validates the interpolation machinery and the dilution floor,
  not predictive skill. Do not quote it as accuracy.
- **Synthetic rich-history test** (fabricated 7-state history, launch
  night → live snapshot → Nov 26, N 638 → 90,000): medians stay sane
  (Chapter 1 1.27%, final mission 0.068%, platinum 0.018%), chain
  monotone, ceilings respected; mean |logit error| vs the week-one prior
  table 1.18 (the prior is the test's own construction, so this measures
  the blend, not the truth). The q05–q95 band covers the prior for
  27/54 trophies — the model hedges between the observed dilution trend
  and the prior's day-3–7 rise rather than copying either.
- **makefun check** (from the kit dir, `--worktree`, no transcripts yet):
  4 passed — `competition_open`, `snapshot` (2 files, 22,973 bytes),
  `license_osi` (MIT), `entrypoint` (main.py → Python 3.13); 1 failed —
  `transcripts_present` (expected: transcripts are bundled at submit
  time by the parent agent); 5 checks deferred to submission
  (repo public / owner / created-after-open / commit in repo /
  duplicate snapshot). Re-run with `--transcripts` after bundling.

## Forecast table

Practice p00 medians are the model's real output on the kit's practice
case (target 2026-11-19 04:30 UTC). The "Synthetic Nov-26 median" column
is the model run on the fabricated rich-history case described above —
an illustration of the final-case behaviour, **not** a prediction to
quote (the real final case is cut at the deadline from the house's
recording).

| id | Trophy | Last rate (launch night) | Practice p00 median (target 04:30 UTC) | Practice q05–q95 | Synthetic Nov-26 median* |
|---:|---|---|---|---|---|
| 1 | Vice Got a Price! (Platinum) | 0.63% | 0.1748% | 0.0338–0.8989% | 0.0178% |
| 2 | Just Like Paradise | 100% | 99.99% | 99.9891–99.99% | 99.99% |
| 3 | Paradise Lost (Ch.1) | 1.25% | 0.3467% | 0.0671–1.7708% | 1.2735% |
| 4 | T**s Ass Drug Money (Ch.2) | 1.1% | 0.3051% | 0.059–1.561% | 0.2084% |
| 5 | Vice City, Baby! (Ch.3) | 1.1% | 0.3051% | 0.059–1.561% | 0.0824% |
| 6 | Top Of The World, Ma (Ch.4) | 1.1% | 0.3051% | 0.059–1.561% | 0.0758% |
| 7 | You Gotta Know When to Get Off (final) | 1.1% | 0.3051% | 0.059–1.561% | 0.0684% |
| 8 | Trust | 1.1% | 0.3051% | 0.059–1.561% | 0.317% |
| 9 | Break the Bank | 1.1% | 0.3051% | 0.059–1.561% | 0.0708% |
| 10 | Family Matters | 1.1% | 0.3051% | 0.059–1.561% | 0.0708% |
| 11 | DWNFALL | 1.1% | 0.3051% | 0.059–1.561% | 0.0708% |
| 12 | Hi-Jacked | 1.1% | 0.3051% | 0.059–1.561% | 0.0708% |
| 13 | Snow Storm | 1.1% | 0.3051% | 0.059–1.561% | 0.0708% |
| 14 | A Long Road (100%) | 0.63% | 0.1748% | 0.0338–0.8989% | 0.0215% |
| 15 | Cunning Stunts (250 stunts) | 0.63% | 0.1748% | 0.0338–0.8989% | 0.0248% |
| 16 | Master of None | 0.78% | 0.2164% | 0.0418–1.1109% | 0.0699% |
| 17 | The Shootist | 0.63% | 0.1748% | 0.0338–0.8989% | 0.0329% |
| 18 | Top Fun | 0.63% | 0.1748% | 0.0338–0.8989% | 0.0248% |
| 19 | Earn Cash, Not Kudos | 0.78% | 0.2164% | 0.0418–1.1109% | 0.0487% |
| 20 | Baited My Tackle | 2.04% | 1.1833% | 0.2306–5.8417% | 2.2691% |
| 21 | Stick Fiddler | 0.94% | 0.2607% | 0.0504–1.3364% | 0.5472% |
| 22 | Road to Rehabilitation | 0.78% | 0.2164% | 0.0418–1.1109% | 0.0434% |
| 23 | This Was No Boat Accident | 0.78% | 0.2164% | 0.0418–1.1109% | 0.0434% |
| 24 | The Lone Rider | 0.78% | 0.2164% | 0.0418–1.1109% | 0.0434% |
| 25 | Strangers & Freaks | 0.63% | 0.1748% | 0.0338–0.8989% | 0.0246% |
| 26 | Supply & Demand | 0.63% | 0.1748% | 0.0338–0.8989% | 0.0304% |
| 27 | Spin On This! | 0.78% | 0.2164% | 0.0418–1.1109% | 0.0434% |
| 28 | Positive Mental Attitude | 0.63% | 0.1748% | 0.0338–0.8989% | 0.0278% |
| 29 | Fair Game | 0.78% | 0.2164% | 0.0418–1.1109% | 0.0403% |
| 30 | Don't Touch My Deck | 0.63% | 0.1748% | 0.0338–0.8989% | 0.0217% |
| 31 | You WAiNKer! | 0.63% | 0.1748% | 0.0338–0.8989% | 0.0217% |
| 32 | One Last Thing | 0.63% | 0.1748% | 0.0338–0.8989% | 0.0248% |
| 33 | Hot Together | 0.94% | 0.2607% | 0.0504–1.3364% | 0.1243% |
| 34 | Retail Therapy | 0.78% | 0.2164% | 0.0418–1.1109% | 0.0681% |
| 35 | Whip It Good | 1.25% | 0.3467% | 0.0671–1.7708% | 0.3589% |
| 36 | Gunslinger | 0.78% | 0.2164% | 0.0418–1.1109% | 0.0826% |
| 37 | Catch Me If You Can | 0.94% | 0.2607% | 0.0504–1.3364% | 0.1816% |
| 38 | Challenging Tank | 1.57% | 0.4355% | 0.0844–2.216% | 0.1656% |
| 39 | Sneakthief | 0.94% | 0.2607% | 0.0504–1.3364% | 0.0732% |
| 40 | Maximum Withdrawal | 0.78% | 0.2164% | 0.0418–1.1109% | 0.0379% |
| 41 | A Classic Blitz Play | 0.94% | 0.2607% | 0.0504–1.3364% | 0.0496% |
| 42 | The Goodtime State | 0.78% | 0.2164% | 0.0418–1.1109% | 0.0547% |
| 43 | Pest Control | 0.78% | 0.2164% | 0.0418–1.1109% | 0.0379% |
| 44 | In The Beginning… | 0.94% | 0.2607% | 0.0504–1.3364% | 0.1142% |
| 45 | Capital Offense | 0.78% | 0.2164% | 0.0418–1.1109% | 0.0519% |
| 46 | Let’s Go Bowling | 0.78% | 0.2164% | 0.0418–1.1109% | 0.1298% |
| 47 | Personal Best | 0.78% | 0.2164% | 0.0418–1.1109% | 0.0454% |
| 48 | Actions Speak Louder | 0.78% | 0.2164% | 0.0418–1.1109% | 0.0588% |
| 49 | Bad Luck, I Guess | 0.78% | 0.2164% | 0.0418–1.1109% | 0.0454% |
| 50 | The Truth Is Out There | 0.78% | 0.2164% | 0.0418–1.1109% | 0.1534% |
| 51 | Ten Pin | 0.78% | 0.2164% | 0.0418–1.1109% | 0.0759% |
| 52 | Dime on the Dollar | 0.78% | 0.2164% | 0.0418–1.1109% | 0.0364% |
| 53 | Treat Yo Self | 0.94% | 0.2607% | 0.0504–1.3364% | 0.0638% |
| 54 | Little Boxes | 0.78% | 0.2164% | 0.0418–1.1109% | 0.0364% |
| 55 | On the Spot | 0.78% | 0.2164% | 0.0418–1.1109% | 0.0494% |

\* Synthetic Nov-26 column: model output on a fabricated evolving history
(launch night → live snapshot → Nov 26, N 638 → 90,000). Illustration only.

## Files

- `main.py` — the entry (only entrypoint; `main.ts` starter removed, the
  rules allow exactly one of the two).
- `README.md` — this file.
- `docs/report.html` — the same forecast as a standalone page with charts.
- `LICENSE` — MIT (kit starter license).

No claim code, no wallet material, no secrets anywhere in this repository.
