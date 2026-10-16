# Learning Who to Introduce
### Thompson Sampling, Global Matching and Decision-Relevant Questions for the One Introduction Problem (Sequential Matching)

| | |
|---|---|
| **Team** | Dynamic Duo |
| **Members** | Sharvakanth A, Kanaka Vikraman C |
| **Event** | Vouchsafe Sequential Matching Hackathon |
| **Submission** | Round 1 Research Submission (see `Dynamic Duo Research Proposal`) |
| **Language / runtime** | Python 3.11, CPU only |

## About the project

The **One Introduction Problem** is a sequential decision problem. Each day the system decides which pairs of members to introduce, learns from the feedback, and tries to maximize **MSMI** (Mutual Second-Meeting Intention) per 100 arrived members over a 60-day episode.

Our central design rule, taken from the Day 3 workshop: **write the hard rules first, encode them so they cannot be broken, and only then let a soft objective choose among what remains.** The system learns what works by acting, since a pair is only ever observed if it is introduced, so it has to be willing to try pairs it is unsure about, while never breaking a stated constraint (for example, a person who will not date a smoker).

## Why the problem is hard

- **Selective feedback:** if A is never introduced to B, we never learn whether they would have worked. A greedy policy can lock onto a poor choice.
- **Two-sided consent:** an introduction works only if A accepts B and B accepts A, so the two directions are modelled separately.
- **Competing choices:** giving one person their best match may block two better matches for others, so we choose across the whole pool at once.
- **Hard rules vs soft goals:** hard rules must never be traded for score; preferences only rank what is feasible.
- **Cold start:** newcomers have no history, so exploring must be cheap and informed by a prior.
- **Drift and delayed feedback:** outcomes take days to arrive and conditions change.

## Approach at a glance

1. **Hard eligibility:** a binary mask B(i, j) removes forbidden pairs before any scoring, so no score can buy its way past a rule. A post-solve audit checks the plan, and violations must be zero.
2. **Bayesian outcome model:** a bootstrap ensemble of logistic regressions (about 10 members in the proposal) predicts P(MSMI); disagreement between members measures uncertainty.
3. **Thompson sampling:** each day one ensemble member is sampled as the "true world" and used to score pairs. Exploration shrinks as evidence builds, with no epsilon to tune.
4. **Global matching with a waiting threshold:** a maximum-weight matching across the pool (each person in at most one pair), with the option to wait if the best plan is not good enough.
5. **Value-of-information clarification:** the daily question budget (12 units; 3 for the hard-constraint bundle, 1 for a soft question) is spent only where an answer could change a decision.

Supporting ideas in the proposal: contextual priors for newcomers, discounting of old evidence, a small random-exploration slice (about 5%) for detecting model errors, and calibration, coverage and feasibility checks.

## Proposal vs. current implementation

The proposal describes the full design. This repository is the working implementation, and some parts are simplified or not yet built:

| Proposal element | Status in this repo |
|---|---|
| Hard eligibility mask + post-solve audit | Implemented (`eligibility_mask.py`); the proposal also covers children and schedule rules, and the code covers gender, age, zone and smoking |
| Bootstrap ensemble outcome model | Implemented: 10 logistic models refit daily on bootstrap resamples of recent outcomes |
| Thompson sampling over the ensemble | Implemented (`thompson_sampling.py`) |
| Global maximum-weight matching | Greedy approximation (`matching.py`), not the exact blossom algorithm |
| Waiting threshold | Implemented (threshold 0.3) |
| Value-of-information clarification | Simplified heuristic (fixed value per field type), not the sampled expected-VOI re-solve |
| Separate A→B and B→A acceptance models | Not yet implemented (single pair-level model) |
| Newcomer priors, discounting, random slice | Placeholders only (ablation flags exist; behavior is the same as `full`) |
| Contested-person plan values (shadow price) | Not implemented |
| Six scenarios x 20 seeds x ablations | Harness exists; the generator does not yet differentiate scenarios (see Known issues) |

---

## Components

The system combines four ideas, and the experiment harness ablates each one to measure its contribution:

| Component | Idea | File |
|---|---|---|
| Hard constraint mask | Never propose pairs that violate gender, age, zone or smoking constraints | `eligibility_mask.py` |
| Outcome ensemble | Bootstrap ensemble of logistic models predicting P(MSMI) per pair | `outcome_model.py` |
| Thompson sampling | Each day, sample one ensemble member as the "true world" and score with it | `thompson_sampling.py` |
| Global matching | Select non-overlapping pairs that maximize total score, with an optional "wait" threshold | `matching.py` |
| Value-of-information clarification | Spend a daily ask budget on the most valuable unobserved fields | `clarification.py` |

---

## 1. Project structure

```
sequential_matching/
├── dev_kit.py              # Simulator + synthetic world generator (do not modify)
├── eligibility_mask.py     # Hard-constraint mask + audit function
├── outcome_model.py        # LogisticRegressionModel + OutcomeEnsemble
├── thompson_sampling.py    # ThompsonSampler (scores all pairs with one sampled model)
├── matching.py             # Greedy max-weight matching, matching with waiting threshold
├── clarification.py        # ValueOfInformationSelector (which questions to ask)
├── ablations_main.py       # Runs one 60-day episode with ablation flags
├── experiment_runner.py    # Runs many episodes in parallel
├── analyze_results.py      # Bootstrap CIs + hypothesis tests (H1, H2, H3, H6)
├── plot_results.py         # Produces comparison/ablation/hypothesis PNG charts
├── test_quick.py           # One-day diagnostic sanity check
├── validate_runtime.py     # Checks one day runs within the 10 s budget
└── requriments.txt         # Python dependencies (note the filename spelling)
```

## 2. Setup

Requires Python 3.11.

```bash
cd sequential_matching
python3.11 -m venv venv
source venv/bin/activate
pip install -r requriments.txt        # file is spelled "requriments.txt"
```

If pinned versions fail to resolve, the unpinned install works:

```bash
pip install numpy pandas scipy scikit-learn matplotlib
```

The core pipeline uses only the standard library; `matplotlib` is needed only for `plot_results.py`.

## 3. Daily pipeline

`run_episode_with_ablations` in `ablations_main.py` runs this loop for each of the 60 days:

1. **Observe** the simulator state (members, past introductions, feedback).
2. **Train** the outcome ensemble on the observed feedback.
3. **Build the mask**: a dict `{(a_id, b_id): 1/0}` marking feasible pairs (hard constraints).
4. **Score** every pair using one ensemble member chosen by Thompson sampling (infeasible pairs score 0).
5. **Filter**: drop pairs already introduced and pairs whose members still hold an active introduction (7-day slot).
6. **Match**: pick non-overlapping pairs, optionally waiting if the best score is below a threshold (0.3).
7. **Clarify**: choose questions to ask using the VOI heuristic (budget 12 cost units per day) and resolve them in the simulator.
8. **Advance** the simulator with the chosen pairs; feedback is generated from a compatibility function.
9. **Audit** the chosen pairs against hard constraints and log any violations.

After day 60, the episode runs 40 more days with no introductions so late feedback can arrive, then collects final metrics.

## 4. Module reference

### `dev_kit.py`
- `generate(seed, n, pool_id, variant)` creates `n` synthetic members as dicts (age 25–65, gender, preferred gender, age range, zone, smoking and children fields, etc.).
- `Simulator` exposes `observe()`, `resolve_asks(asks)`, `advance(pairs)` and `metrics()`.
- Ask costs: `constraints` = 3, every other field = 1; total budget 1000.
- `metrics()` returns `total_introductions`, `mutual_second_meeting_intention`, `msmi_per_100_arrived_members` and `ask_cost`.

### `eligibility_mask.py`
- `build_eligibility_mask(members)` returns the feasibility mask over all unique pairs. It checks gender preference, age range (both directions), acceptable zones and smoking compatibility.
- `check_hard_constraints(state, pairs)` audits proposed pairs and returns `(is_valid, violations)`.

### `outcome_model.py`
- `LogisticRegressionModel`: features are age difference, gender match, goal compatibility, lifestyle compatibility, zone overlap and smoking compatibility, combined through a sigmoid.
- `OutcomeEnsemble(n_ensemble, seed)`: `predict_msmi(a, b, idx)` for one member; `predict_msmi_ensemble(a, b)` for mean and std; `get_uncertainty(a, b)`.

### `thompson_sampling.py`
- `ThompsonSampler.sample_world()` picks an ensemble index for the day.
- `score_all_pairs(members, mask, idx)` returns `{(a_id, b_id): score}` clamped to [0, 1].

### `matching.py`
- `greedy_max_weight_matching(scores, members_by_id)` sorts pairs by score and greedily takes pairs whose members are both unmatched.
- `matching_with_waiting_threshold(scores, members_by_id, threshold)` returns `(pairs, should_wait, total_score)`; if the top score is below the threshold it returns nothing and waits.

### `clarification.py`
- `ValueOfInformationSelector.select_asks(state, max_cost=12)` builds candidate asks for each unobserved field and ranks them by value per cost (constraints: value 2.0, cost 3; soft fields: value 1.0, cost 1). It then fills the budget greedily.

## 5. Running experiments

**Sanity check and timing**

```bash
python test_quick.py          # one-day diagnostic with score distribution
python validate_runtime.py    # verifies a day runs well under the 10 s budget
```

**Single episode**

```bash
python ablations_main.py --variant development --seed 42 --ablation full
python ablations_main.py --ablation greedy_ranking --verbose
# output goes to results/<ablation>/<variant>/<variant>_<seed>.json
```

Arguments: `--variant`, `--seed`, `--n` (pool size, default 200), `--days` (default 60), `--ablation`, `--output_root` (default `results`), `--verbose`.

**Batch run**

```bash
python experiment_runner.py --small      # 2 scenarios x 2 seeds x 9 ablations = 36 episodes
python experiment_runner.py --seeds 5    # all 6 scenarios x 5 seeds (270 episodes, about 6 min on 4 workers)
python experiment_runner.py --workers 4  # full: 6 scenarios x 20 seeds x 9 ablations = 1,080 episodes
```

**Analysis and charts**

```bash
python analyze_results.py   # writes results/analysis.json
python plot_results.py      # writes results/comparison.png, ablations.png, hypotheses.png
```

## 6. Ablations

| Ablation | What changes |
|---|---|
| `full` | All components on |
| `no_clarification` | No questions asked |
| `random_clarification` | 4 random members asked about `constraints` instead of VOI selection |
| `greedy_ranking` | Greedy matching without the waiting threshold (no hard-rule exemption: score-0 pairs are still never chosen) |
| `epsilon_greedy` | Always scores with ensemble member 0 instead of Thompson sampling |
| `soft_penalty_mask` | All pairs treated as feasible (no hard mask) |
| `no_waiting_threshold` | Matches regardless of score |
| `no_discounting` | Trains on all history instead of the 400 most recent outcomes |
| `no_random_slice` | Placeholder (identical to `full`) |

`no_thompson` is accepted as an alias of `epsilon_greedy`. All ablations obey the 7-day single-slot rule.

## 7. Hypotheses tested (`analyze_results.py`)

- **H1**: Thompson sampling beats epsilon-greedy, with the largest gap in `cold_start`.
- **H2**: Global matching beats greedy ranking, with the smallest gap in `sparse`.
- **H3**: VOI clarification beats both no clarification and random clarification.
- **H4**: Discounting old evidence improves performance under shifted outcome weights and changing response conditions. *(proposal only; not yet in `analyze_results.py`)*
- **H5**: Newcomers reach their first introduction and first MSMI sooner under the full method than under greedy selection, without hurting established users. *(proposal only; not yet in `analyze_results.py`)*
- **H6**: The hard mask gives zero violations and MSMI at least as high as the soft penalty.

Each uses a 95% bootstrap confidence interval (1000 resamples) over per-episode MSMI.

## 8. Output format

Each episode writes a JSON file containing the config, the ablation flags, `daily_logs` (pairs introduced, total score, asks made, rule violations, whether it waited) and `final_metrics`.

## 9. Debugging log and known limitations

### Fixed (the "0.0 MSMI" investigation)

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | Analysis printed MSMI 0.0 | `analyze_results.py` loaded scenarios that had not been run (e.g. `cold_start` in a `--small` sweep), and `bootstrap_ci([])` returned 0, so missing data looked like a real 0.0 | Missing data is now `n/a`; hypotheses with no data are skipped; H1 falls back to `development` if `cold_start` is absent |
| 2 | `analyze_results.py` crashed with `TypeError` | `len()` called on the integer `rule_violations` | Counts handled as int (or list) |
| 3 | Charts showed only `full` / "No results found" | `plot_results.py` read `results/episodes/...` but the runner writes `results/<ablation>/<variant>/`; it also expected `no_thompson` | One layout everywhere; `no_thompson` is an alias of `epsilon_greedy`; shared `ABLATIONS` list in `ablations_main.py` |
| 4 | `greedy_ranking` and `no_waiting_threshold` broke hard rules (about 2,300 violations per episode) | Infeasible pairs have score 0, and the greedy matcher still paired people at score 0 | `matching.py` never selects a pair with score <= 0 |
| 5 | Same ~100 pairs re-introduced every day (6,000 introductions among 200 people); `full` MSMI of 47 | Nothing excluded already-introduced pairs, and feedback is stored per pair, so repeats overwrote each other | A pair is introduced at most once |
| 6 | Everyone introduced every day | The 7-day single-slot rule from the proposal (Sec. 4.1) was not enforced | `ablations_main.py` enforces one active introduction per person for 7 days |
| 7 | Ensemble never learned | `OutcomeEnsemble.train` drew bootstrap indices but never fitted anything | Real bootstrap logistic fit on recent outcomes (L2 pull to the prior) |
| 8 | Results shifted when code order changed | Global `random.seed(...)` was called inside the model, sampler and clarifier, which also reseeded the simulator's feedback draws | All modules use local `random.Random` instances |
| 9 | `validate_runtime.py` always waited | `threshold=8.5` on scores in [0, 1] | Threshold 0.3 |
| 10 | `requriments.txt` failed to install | Stray `EOF` line | Remove it / rename to `requirements.txt` |

### Remaining limitations (read before interpreting results)

1. **`generate()` ignores `variant`** (and `dev_kit.py` is marked do-not-modify), so all six scenarios are the same world. Cross-scenario comparisons and H1/H2's scenario claims are not meaningful yet.
2. **Several ablations are identical to `full` in `dev_kit`.** The simulator's feedback does not depend on clarification answers, and our scores never fall below the 0.3 waiting threshold, so `no_clarification`, `random_clarification`, `greedy_ranking` and `no_waiting_threshold` produce the same episodes as `full`. This is a property of the toy simulator, not a bug in the ablation flags.
3. **Soft penalty scores higher MSMI than the hard mask.** The simulator's `_compute_msmi` ignores gender, so introducing people outside their stated preferences is not penalised there. The hard mask still gives 0 violations against about 600 per episode for the soft ablation, so the violations half of H6 passes and the MSMI half does not.
4. **Matching is greedy**, not exact maximum-weight (blossom) matching.
5. **Clarification** uses fixed values per field, not an expected-VOI re-solve.
6. **Not implemented:** separate A-to-B and B-to-A acceptance models, newcomer priors, the random-exploration slice (`no_random_slice` is a placeholder), contested-person shadow prices.
7. **MSMI per 100** is `mutual intentions / pool size x 100`, so it can exceed 100 when each person has several introductions over 60 days.

## 10. Reproducibility

**Environment (as used in our runs)**

- Python 3.11 (macOS, Apple Silicon), CPU only, no GPU, no network calls at run time.
- Installed versions: numpy 2.4.6, pandas 3.0.6, scipy 1.17.1, scikit-learn 1.9.1, matplotlib 3.11.2.
- The core pipeline uses only the standard library; the packages above are needed for plotting and analysis.
- To freeze the exact environment after a successful run:

```bash
pip freeze > requirements.lock.txt
```

**Seeds**

- Every episode is fully specified by `(variant, seed, ablation)`. The default development seed is `42`; the full sweep uses seeds `42` to `61` (20 seeds), and `--small` uses `42` and `43`.
- The same seed is passed to the world generator, the ensemble (`seed + i` per member), the Thompson sampler (`seed + day + 7919`), and the random-clarification ablation (`seed + day + 12345`).
- All methods and ablations are run on identical seeds, so differences can be attributed to the method.

**Reproducing a result**

```bash
# one episode
python ablations_main.py --variant development --seed 42 --ablation full

# the full experiment sweep, then analysis and charts
python experiment_runner.py --workers 4
python analyze_results.py
python plot_results.py
```

Each episode's JSON log records the variant, seed, ablation, pool size, number of days and the ablation flags, so any result file can be traced back to the exact command that produced it.

**Container build (planned for Day 7)**

A minimal `Dockerfile` for a clean, pinned run:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.lock.txt .
RUN pip install --no-cache-dir -r requirements.lock.txt
COPY . .
CMD ["python", "ablations_main.py", "--variant", "development", "--seed", "42", "--ablation", "full"]
```

```bash
docker build -t sequential-matching .
docker run --rm sequential-matching
```

**Execution constraints (from the proposal)**

| Constraint | Limit | How we check |
|---|---|---|
| Daily decision time | 10 s including inference, decisions and serialization | `python validate_runtime.py` (measured about 0.03 s per day at 200 members) |
| Memory | 1 GB, no need to store the full training set at inference time | To be measured before submission |
| Hardware | CPU only, no external services | Standard-library core, no network calls |
| Hard-rule violations | Must be zero for our method | Logged per day as `rule_violations` and audited after every solve |

**Frozen submission record** (fill in at submission)

| Item | Value |
|---|---|
| Commit SHA | `<to be filled at freeze>` |
| Python version | 3.11 |
| Dependency lock file | `requirements.lock.txt` |
| Date frozen | `<to be filled>` |

**Known reproducibility caveats**

- Only `dev_kit.py` uses Python's global `random` (world generation and feedback draws). Our modules use local `random.Random` instances, so adding or reordering our code does not change the simulator's random stream.
- `experiment_runner.py` runs episodes in separate worker processes, so each process has its own random state. Output order varies between runs, but the contents of each episode file should not.
- Re-running the same `(variant, seed, ablation)` gives the same JSON; verify this after any code change.

## 11. Suggested next steps

- Make `generate()` honour `variant` so each scenario differs.
- Unify result directories and ablation names across all scripts.
- Re-run `experiment_runner.py --small`, then the full 1,080-episode sweep, and regenerate the charts.
