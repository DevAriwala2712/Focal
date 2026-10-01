# F9 — adversarial verification of the fix wave (F1, F2, F3, F5, F6, F7)

Replaces `experiments/results/x10.json` / `x10_REPORT.md` (INVALIDATED, **B13**: the previous verifier
skipped the required clean-clone rerun, missed B1–B12, and miscounted its own keep tally).

- **Worktree**: `.claude/worktrees/fix-v2-f9`, branch `task/f9`, HEAD at start `11de65f`
- **`config_sha256`**: `18cce7b1ea88855bc054b42e6cbe3219eef73b2909eb091d07d5ada3df170970`
  — computed by F9 with `hashlib.sha256(open('configs/fix.yaml','rb').read())`, not copied from any
  `f*.json` and not taken from the task brief. It matches all six result files.
- **Machine-readable**: `experiments/results/f9.json`

| | |
|---|---|
| Claims audited | **49** |
| **keep** | **43** |
| **downgrade** | **5** |
| **drop** | **1** |
| Mandatory checks run | 9 of 9, per task |
| Statuses confirmed | F1 PASS · F2 FAIL · F3 PASS · F5 PASS-with-caveat · F6 PASS · F7 FAIL — **all six self-reports are correct** |

The tally above is **computed from the claims list** by `experiments/f9_emit.py`, never typed by hand —
A10 was invalidated partly for asserting 3/5 KEEP while listing 2, and F9 makes that arithmetic
impossible to get wrong (`summary.counts_sum_to_total: true`).

---

## Bottom line

**The fix wave is materially sound and materially honest.** F9 went looking specifically for the things
that killed the x-wave — a fabricated number, a test that compares a quantity with itself, a threshold
that is loaded but never applied, a threshold moved after seeing a result — and **found none of them**.
Every arithmetic claim F9 recomputed reproduced, several to the last floating-point digit.

The scientific bottom line is **negative, and correctly reported**: gate v2 does not meet its
false-alarm rule, and sub-pixel allocation *hurts* once the block fraction has to be estimated from
Sentinel-2 rather than read off the truth. The experiments say so themselves, prominently.

What F9 did find: one **false factual assertion** about a hash (dropped), one **over-claimed deviation
mechanism** that F9 measured and disproved (downgraded), **two stale hashes** in report headers
(downgraded), and **two F7 numbers** that are arithmetically exact but must not be read as detection
quality (downgraded).

### External artifacts used (not in git history)
`data` → shared Sentinel-2 / experiments cache, and `models` → pretrained SEN2SR-lite weights, both
symlinks. Both are legitimate; `data` is genuinely required, since F7's block-sum tests read
`data/experiments-cache/f7_production_state.npz`. A **third** external dependency is worth recording: the
worktree has no `.venv`, so the parent repository's `/Users/devariwala/Desktop/Focal/.venv`
(Python 3.11.15; numpy/torch/scipy, **no pandas**) was used.

---

## The one DROP, and why it is not a downgrade

> **F1_REPORT.md** asserts the hash `7ee83389827340ed4e40b03917a7278d6bad11e2a6dac68460e8596c0111e7e6`
> is *"(66 hex chars)"* and *"cannot be a valid SHA-256 digest"*.

Both halves are false. `len()` returns **64**, the string is valid hex — and it is the **exact sha256 of
`configs/fix.yaml` at the F0 pre-registration commit `020880f`**, i.e. precisely the hash F6 legitimately
computed before the YAML fix landed.

A downgrade is for a claim that is true but weaker than stated. There is no weaker true version of this
one to retreat to, so it is **dropped**. It is also the single place in the wave where a quantitative
assertion was made by *inspecting* a string rather than *measuring* it — the practice
`f9_checks.arithmetic_checked_by: code, not prose` exists to forbid.

**Consequence: none scientific.** F1's own `config_sha256` is correct and independently verified. Only the
paragraph dismissing the other hash needs to go.

---

## The 9 mandatory checks

### 1. `point_estimate_inside_its_ci` — PASS (F1, F2, F3, F5, F6, F7)

`experiments/f9_check_ci.py` walks every `f*.json`, finds every (point estimate, CI) pair under each
convention actually used, and evaluates `lo <= point <= hi` in code.

| task | CI pairs found | outside own CI |
|---|---|---|
| F1 | 76 | **0** |
| F2 | 34 | **0** |
| F3 | 15 | **0** |
| F5 | 40 | **0** |
| F6 | 9 | **0** |
| F7 | 1 | **0** |
| **total** | **175** | **0** |

A first pass flagged one apparent violation in `f3.json`'s `keep_rule`. It was **F9's own pairing
heuristic**: that node carries *two* point estimates (normalised and un-normalised coverage) and the bare
`coverage` was being matched against `ci95_without_normalisation`. Paired correctly, both sit inside their
own CIs. F9 fixed the scanner rather than reporting a false finding. This is the exact **B4** defect that
invalidated x3, and it does not recur anywhere in the fix wave.

### 2. `no_placebo_code_path_reads_post_event_date` — PASS

F9 did **not** read the guard and reason about it. `experiments/f9_try_break_guard.py` **attempts 16
loads** of the post-event date through every placebo/calibration entry point, using the same real config
loader the experiments use. All 16 behaved correctly:

- `guarded_load_date` raised on `2024-12-06`, `2024-07-30`, and a `datetime.date` object;
- `build_sr_fold` raised with the post date as `held_out` **and** smuggled into `pre_dates`;
- `build_10m_fold` raised both ways;
- **`f2_gate_v2.build` raised with the post date monkeypatched into `SR_POOL`** — this is the function
  F7's `recompute_tau` calls, so the τ/calibration path is covered;
- malformed spellings (`2024-7-30`, `20241206`, `2024/12/06`, `9999-01-01`) **fail closed**;
- controls `2024-07-29` and `2024-01-16` still load, so the guard is not merely always-raise.

**Bypass audit.** Two *unguarded* dihedral loaders exist (`placebo_v2._load_dihedral`,
`f2_gate_v2.load_dihedral` — raw `np.load` on a date-templated path). They are **provably unreachable**
with a post date, because in every caller the guarded 10 m load over the same date list executes first and
raises; F9 confirmed this by attack, not by reading. Worth hardening; not a live defect.

F7 loads the post date **deliberately** for the production map by a separate, documented route. That is
correct, and τ never sees it.

### 3. `every_loaded_threshold_is_used_mutation_test` — PASS (no unused parameter anywhere)

For each threshold F9 ran the mutation test, then **sabotaged the line that applies it** and confirmed the
test now fails. Every sabotage was reverted and `git diff --stat` confirmed empty.

| threshold | sabotage | result |
|---|---|---|
| **τ** (F2/F7) | `win_detected = win_valid & (win_scores > tau)` → `win_valid & isfinite(win_scores)` (x9's **B10** bug) | both `TestTauMutation` tests **FAILED** → test is not vacuous |
| **λ** (F2/F7) | forced the `lam == 0` early return always | `test_lambda_changes_which_sub_pixels_are_chosen_but_never_how_many` **FAILED** |

τ is additionally a **required positional argument** with no default; `None`/`NaN`/`inf` raise. On the real
production map τ×0.5 / τ / τ×2 give **287,477 / 256,770 / 123,803** flagged px — strictly monotone, and the
fitted-τ count equals the headline mapped-change area exactly.

**λ = 0.0 is not an unused loaded parameter**, so the stop rule does not fire. λ is read from `f2.json`
(F7 line 578) and passed into every `apply_gate_v2` call (lines 630/651/660/679); its mutation test is
sabotage-verified; and 0.0 is a *fitted* optimum from a real 6-point grid search on a real objective
(split-half dihedral Jaccard, highest at 0 and falling monotonically). The consequence to carry forward is
scientific, not procedural: **the neighbour-agreement term contributes nothing to any reported result**, so
no claim that it helps is supported. F2 says exactly this.

> **Finding.** F7's own `test_tau_is_used_not_merely_loaded` reads those three numbers back *out of
> f7.json* rather than re-running the gate, so it cannot catch a code regression. The claim stands on the
> `trustsr`-level mutation tests instead, which F9 sabotage-verified.

### 4. `no_test_compares_quantity_with_itself` — PASS (the direct B7 regression check)

x4's `check_block_sum_consistency` compared `h*w*100 m²` with `h*w*100 m²`. F2/F7's replacement derives
the two sides independently: **observed** by counting `CORE|ALLOCATED` px in the 4×4 footprint of the
**output class map**, **expected** by recomputing `round(16f)` from the unmixing fraction. The allocator's
`n_per_block` is not a parameter, and a signature test asserts that.

F9 proved non-tautology by **sabotage, not inspection**: rewriting `observed` into a second copy of the
`expected` formula — recreating B7 exactly — made **both** anti-tautology tests fail, including the one
running on the **real production map**:

```
FAILED tests/test_gate_v2.py::TestBlockSum::test_the_check_is_not_tautological_it_can_fail
FAILED tests/test_f7_wayanad_v2.py::test_block_sum_on_the_real_map_is_not_tautological
```

Also verified: F1's window FAR takes each gate's **own** flag map as an explicit argument (B3's fix); F5's
two `alloc_*` methods receive the *identical* fraction and allocator but rank by two independently built
fields (bilinear upsample of the 10 m index vs. the real SR forward-pass index) — a self-comparison would
return exactly 0.0, and the measured delta is non-zero on 87 of 118 images.

**F3's `median(d) = 0` is a genuine tautology** — `offset_held` is *defined* as the median residual, so
removing it zeroes the median identically. The stop rule does **not** fire, because F3 names it as
definitional in its own limitations and explicitly declines to use it as evidence, resting its keep rule on
**coverage** instead (σ comes from the pool's across-date and dihedral variance, never from `d`). Had F3
offered it as evidence, that claim would have been **DROPPED**.

### 5. `every_result_carries_fix_yaml_hash` — PASS in all six JSONs; FAIL in two reports

All six `f*.json` carry `18cce7b1…f170970`, byte-for-byte equal to F9's own computation.

**Full lineage, verified commit by commit:**

| commit | `configs/fix.yaml` sha256 | parses? |
|---|---|---|
| `020880f` F0 pre-registration | `7ee83389…0111e7e6` | **no** — `yaml.safe_load` raises `ParserError` |
| `298e214` F3's YAML fix | `192345cd…d5a905b3` | yes |
| `0da15a3` F1's commented fix (= merged branch) | `18cce7b1…df170970` | yes |

`yaml.safe_load(298e214) == yaml.safe_load(0da15a3)` is **True** — the two independent fixes differ only by
a comment. The F0 original genuinely does not parse, which **validates the reason for the fix**. No check
name, threshold or semantic changed.

**Both "F0 amend" commits are honest.** F9 read the full diffs, not the messages: `6d2f722` (f6.json) and
`0059c2f` (f3.json) each change **exactly two lines** — the `config_sha256` value and a new
`config_sha256_note` explaining why. **No other field was touched in either commit.**

> **But the reports were not amended with the JSONs.** `f3_REPORT.md` still prints the stale (and
> truncated) `192345cd156583c6...` and `f6_REPORT.md` still prints the stale `7ee83389…0111e7e6`. For those
> two tasks the human-readable and machine-readable records disagree. Downgraded, not dropped: each was the
> true hash of the file that task actually read, and the variants are semantically identical.

**Artefact chain also verified:** `f1_calibration_scores.npz` recomputes to
`53a2fe8a…fd2679e4`, matching **both** F1's export claim and F2's consumption claim — the F1 → F2 hand-off
is verified end to end.

### 6. `gates_in_a_table_differ_in_code` — PASS (no B2 regression)

The historical bug was x3's `gate_v1_with_a5_sigma` and `gate_v2_placeholder` both being bound to
`gate_v1_trust` — three table rows, one function. F9 compared all five `GATES` entries' `__code__` objects
pairwise: **5 names, 5 distinct code objects, 0 pairs sharing one.**

F1 claims `gate_v1` and `gate_v1_with_a5_sigma` are numerically **identical on the real crop by design**,
not by aliasing. F9 **verified that claim**, and the numbers confirm it — both have identical pixel
numerator **1792**:

- F9 built the fixture in which **only `sigma_a5`** is degenerate: the two gates **diverge by 2,048 px**
  (4,096 vs 2,048 flagged);
- the **mirrored** fixture (only `sigma_v1` degenerate) flips the asymmetry, proving each function reads
  its **own** sigma field;
- on all-finite data with sigmas differing **3×**, both equal the parent mask exactly — independently
  reproducing F1's stated mechanism (`flagged = OBSERVED ∪ INFERRED = parent` wherever `d` and σ are
  finite).

> **Finding.** F1's *own* shipped fixture NaNs **both** sigmas in the same quadrant, so it separates
> `rule_10m` from `gate_v1` but does **not** separate `gate_v1` from `gate_v1_with_a5_sigma`. F9 supplied
> the missing fixture. F1's test suite has that gap; the claim itself survives.

Also noted: F1's table row `gate_v2 = BLOCKED` was correct at F1's run but is now **superseded** — after F2
landed, `placebo_v2.gate_v2` is the real gate. `f1_REPORT.md` still describes it as a stub. F10 must not
reprint that row as current.

### 7. `denominators_stated` — PASS (one documentation gap, in F3)

Scripted two ways: require an *n*-like sibling for every CI-bearing JSON node, and scan every report for a
percentage or ratio whose line **and** enclosing table header carry no *n*.

F1 states both pixel *n* and window *n* on every FAR row. F2 reports every rate as `(num X / den Y, Z
blocks)`. F5 and F6 carry an "n images" column on every table. F7 gives pixels/m²/km² with an explicit
denominator line (5,242,880 px in the crop, 5,036,924 valid) **and** an instruction to take percentages
against valid px rather than the total. No bare, unattributable ratio was found anywhere.

**F3 gap:** `f3.json` carries `n_px` on every pooled and per-fold node and the report states "2,358,286
stable px per fold" in prose immediately above the headline table — but the **headline coverage table
itself has no *n* column**, and 4 summary nodes duplicate a number whose *n* lives in the detailed node.
Documentation, not a defect.

### 8. `no_row_compares_classes_with_different_definitions` — PASS

F9 invoked F7's guard directly. It is an **allow-list (default deny)** and accepted only the two registered
pairs, raising `ValueError("refusing to compare")` on everything else:

| pair | result |
|---|---|
| `v1:OBSERVED|INFERRED` vs `v2:CORE|ALLOCATED` | accepted (registered, same quantity) |
| `v1:NO_DATA` vs `v2:NO_DATA` | accepted (registered, identical construction) |
| `v1:UNSUPPORTED` vs `v2:UNSUPPORTED` | **RAISED** — different gating predicate |
| `v1:OBSERVED` vs `v2:CORE` | **RAISED** |
| `v1:INFERRED` vs `v2:ALLOCATED` | **RAISED** |
| `v1:NO_CHANGE` vs `v2:NO_CHANGE` | **RAISED** — not registered |
| `v1:NO_DATA` vs `v1:NO_DATA` | **RAISED** — refuses a self-pair too |

`compare_sets` — the only route to a v1–v2 number — also raised, and a monkeypatch test proves it always
calls the guard. **UNSUPPORTED was correctly excluded:** `f7.json` holds no difference, ratio or IoU
between the two UNSUPPORTED counts, and a test asserts those keys are absent. Both are reported separately
with their definitions.

### 9. `negative_results_present_in_claim_table` — PASS (F10 will have the material)

F9 located each required negative and recorded **how early** it appears, so "present" cannot mean "buried".

| task | negative | position |
|---|---|---|
| F2 | `Status: FAIL` | **char 70** (first line) |
| F2 | window FAR 0.0652, "NOT MET" | Verdict section |
| F5 | NAIP dependence | 30% in, under an explicit *"Reading this honestly"* heading |
| F5 | null without NAIP | 37% in |
| F5 | realistic fraction −0.313 IoU | own heading: *"allocation is worse than not allocating at all"* |
| F6 | B14 named | **char 36** |
| F6 | venus −0.559, pooled −0.097 | first half |
| F7 | `Status: FAIL` | **char 66** |
| F7 | 0.0652 | **char 356**, in a blockquote above every number |
| F7 | F5's −0.313 as a prior against its own map | 7% in |
| F7 | τ/n_pre mismatch | 34% in |

F7 is the strongest: the failure is in the first line, a blockquote warning sits above all numbers, the
figure caption carries it, and F7's own test asserts the report states it within the first 1,500
characters.

---

## Interrogating the deviations: pre-hoc, or making a failure look better?

This is where a rigorous verifier earns its keep, so F9 **measured** both "it made things harder" claims
rather than accepting them.

### τ recomputation (F2) — claim **VERIFIED**, and it is the harder choice

`configs/fix.yaml` says take τ from F1's npz. F1 exported `max(d/σ_v1)` — a different score function,
spanning −25.86 to 255.60, whereas gate v2's own window scores are order 1–10.

F9 recomputed the verbatim τ from the npz: **119.90589141845703**, matching `f2.json`'s
`tau_if_taken_verbatim_from_this_file` to the last digit. Applying it would flag **~nothing** → window
FAR = 0 → **a vacuous PASS** of the ≤ 0.05 rule. Recomputing gave 0.0652 → **FAIL**.

**The deviation is unambiguously the harder choice, and it is the choice that cost F2 its keep rule** —
the opposite of post-hoc rationalisation.

### `e_b` NDVI ceiling 0.20 → 0.30 (F2) — claim **NOT ESTABLISHED** ← downgrade

F2 and the ADR argue: a shorter mixing line inflates `f`, therefore inflates the score, therefore raises
FAR, therefore harder. F9 held the mixing-line **direction** fixed and shrank its **length**:

| line length | mean `f` | mean `round(16f)` | mean score `s` | max `s` |
|---|---|---|---|---|
| 1.00 | 0.395 | 6.31 | **20.36** | 51.14 |
| 0.75 | 0.524 | 8.40 | 20.26 | 47.60 |
| 0.50 | 0.674 | 10.82 | 17.39 | 31.80 |
| 0.25 | 0.821 | 13.20 | **10.59** | 15.90 |

`f` inflates **2.08×** — that part holds. But the **detection score `s = f/σ_f` FALLS, it does not rise**,
because `σ_f` carries its own `1/|e_v − e_b|` factor that more than offsets the inflation of `f`. The chain
breaks at the third link: **a shorter mixing line makes the conformal detector fire *less* readily, not
more.** And since τ is itself recomputed conformally from the same score, the calibration exceedance rate
is pinned near α whatever `e_b` is, leaving the net test-split effect **indeterminate**.

**This is not evidence of gaming.** The escalation was disclosed **pre-hoc**, with the full 0.20–0.50
ceiling table published, and **the keep rule failed anyway** — no pass was manufactured. Only the stated
mechanism is wrong, and no number depends on it. But the *"made it HARDER"* wording must be corrected to
*"direction not established"* before F10 repeats it.

### τ/n_pre mismatch (F7) — claim **VERIFIED**; the headline FAR is a **lower bound**

`σ_f = sqrt(Var(a)(1/n_pre + 1))`, so `σ_f(2)/σ_f(3) = sqrt(1.5/1.3333) = **1.0606601717798212**` — F9
reproduced exactly the ratio `f7.json` reports. The production score is ~6.07% **higher** than at
calibration, so the gate is **more** trigger-happy than τ was calibrated for, and the true FAR is **above**
the already-failing 0.0652. F7 measures this, declines to correct it, and headlines the more favourable
number while disclosing the direction.

**F10 must always present 0.0652 as a lower bound on this gate's true FAR, never as the FAR.**

---

## Reproducibility

### Test suites rerun in this fresh worktree

```
pytest tests/test_f1_placebo.py tests/test_f3_noise_v2.py tests/test_gate_v2.py \
       tests/test_f5_a8_mapping.py tests/test_alloc.py tests/test_f7_wayanad_v2.py \
       tests/test_f6_a2_report.py -v
→ 114 passed in 1.08 s, 0 failed, 0 skipped
```

These are fast synthetic-fixture unit tests, plus three that recompute from the cached F7 production state.
They are **not** a re-execution of the experiments — see *what would change these verdicts*.

**Full repo suite: 329 passed, 1 failed, 1 skipped.** The single failure,
`tests/test_placebo.py::TestFARPixel::test_far_pixel_basic` (asserts `estimate > 0.4`, gets `0.0`), is in
the **quarantined legacy** `trustsr/placebo.py`, which the fix wave never modified — `git log` over
`49a135f..11de65f` for that module and test is empty, so it is **pre-existing** and touches no F1–F7 claim.

### Independent recomputations (different method, not the tasks' own scripts)

- **F6 pooled PSNR delta** — recomputed from `x2.json` per-image records by **three** routes that do not
  use `f6_a2_report.py` (vectorised numpy, a longhand Python loop over the raw JSON, and a per-dataset
  weighted recombination), agreeing to **< 1e-12** at `-0.09707162370791389`. F9's **own separately
  written** paired bootstrap reproduced `[-0.1574637956167713, -0.03696577159967953]`, and a
  normal-approximation CI brackets it. Per-dataset (all 5), NAIP-excluded (`-0.275`, n=116) and all ten
  `ha_metric` means also reproduced.
- **F5 headline** — rebuilt the paired delta from per-image IoU records with F9's own bootstrap:
  all-datasets **+0.008078** `[+0.00347, +0.01410]`, n=118, 87 wins (report +0.0081); excluding NAIP
  **+0.001592** `[-0.00261, +0.00532]`, n=56, 30 wins, **CI spans zero** (report +0.0016).
- **F1/F2 FARs** — every FAR recomputed as numerator/denominator, matched to 1e-6, including
  gate_v2 window `0.065239551 = 64/981` and pixel `0.003639424 = 13167/3617880`.
- **F3 coverage** — `0.9418016423231794 = 6663111/7074856` and `0.8579530099269865 = 6069894/7074856`
  exactly; `gap_pp` exact; pooled shown equal to the per-fold *n*-weighted mean; fold *n* shown to sum to
  the pooled *n*.
- **F7 areas and IoU** — every `m² = px × 6.25` and `km² = m²/1e6` verified; the five class counts sum to
  **5,242,880 = 2560 × 2048** exactly; mapped change = CORE + ALLOCATED = **256,770 px = 1.6048125 km²**;
  v1-vs-v2 IoU recomputed from the intersection as `0.24481761048116252`, matching exactly.
- **F2 τ order statistic** — `k = ceil((n+1)(1-α)) = ceil(982 × 0.95) = 933` for n = 981, consistent with
  the reported `T_(933)`.

### An extra robustness check F9 ran unprompted

**Both** F2 scoring arms fail the window-FAR keep rule on the test split: the F3-normalised arm at
`0.0652 = 64/981` **and** the un-normalised comparator at `0.0540 = 53/981`. So F2's choice to headline the
*worse* arm (per `fix.yaml`'s `interaction_test`) does not change the verdict — **the FAIL is robust to the
arm choice.** Meanwhile the **calibration**-split rate is `0.04893` in *both* arms, i.e. ≤ α, exactly as
split conformal guarantees by construction.

**This is the substantive scientific finding of the wave:** gate v2 is properly calibrated and simply
**does not generalise across the spatial split** — three folds sharing one crop with an interleaved
checkerboard, precisely the situation `fix.yaml`'s pre-registered `units.window_caveat` warned about. The
failure is a real exchangeability gap, **not** an arithmetic slip.

---

## Cross-cutting findings (11 recorded in `f9.json`)

| # | finding | severity |
|---|---|---|
| 1 | Config-hash amendments are honest, but the **reports** were not amended with the JSONs | low (docs) |
| 2 | `fix.yaml` as pre-registered **did not parse**; the two independent fixes are semantically identical | informational |
| 3 | `f1_REPORT.md` contains a **false factual assertion** about a hash, reached by eyeballing | low, process signal |
| 4 | F2's *"the `e_b` escalation made it harder"* is **not established** | **medium** |
| 5 | The τ/n_pre mismatch makes F7's headline FAR a **lower bound** | **medium** |
| 6 | Gate v2 is calibrated but **does not generalise** across the split; both arms fail | **high** |
| 7 | λ = 0 ⇒ neighbour agreement contributes **nothing** to any reported result | low |
| 8 | Several F7 tests assert properties of `f7.json` rather than re-executing code | low (hygiene) |
| 9 | Two **unguarded dihedral loaders**, currently unreachable with a post date | low, latent |
| 10 | The block-sum PASS is a **plumbing invariant**, not a scientific result | informational |
| 11 | One **pre-existing** test failure on the branch, outside the fix wave | low |

---

## Top risks for F10

1. **Quoting F7's 1.60 km² mapped-change area as a detection.** It is what a gate that *failed* its FAR
   rule produces, and the true FAR is worse than measured.
2. **Quoting F5's +0.0081 headline without "NAIP included" attached.** Without NAIP — the one dataset with
   an unexcludable train/test overlap — the effect is indistinguishable from zero at n=56.
3. **Repeating the ADR's claim that the `e_b` escalation made the keep rule harder.** F9 measured the
   mechanism; it does not hold.
4. **Presenting "block-sum deviation 0" as evidence of mapping quality.** It is a plumbing invariant;
   F5's −0.313 IoU is the quality evidence, and it is negative.
5. **Citing the stale config hashes** still printed in `f3_REPORT.md` and `f6_REPORT.md`.

## What would change these verdicts

F9 reran test suites, attacked guards, sabotaged thresholds and recomputed from stored artefacts — but it
did **not** regenerate the SR stack or the placebo folds from imagery. Two things would strengthen or
overturn this audit:

1. **A real end-to-end re-execution of F1/F2/F7** from Sentinel-2 imagery.
2. **An independent check of the cached E1/E8 dihedral product itself**, which every SR-dependent number in
   F1, F2 and F7 inherits without re-deriving. F7's reproduction check confirms this run *matches the
   cache* (max abs diff 0.0) — it does not independently validate the cache.

---

## Verdict

**43 keep · 5 downgrade · 1 drop**, across 49 claims — and the stop rule
(*"a tautological test or an unused loaded parameter → that claim is DROPPED"*) **did not fire**, because
F9 searched specifically for both triggers and found neither. Every candidate was tested by **sabotage**
rather than inspection: both block-sum tests fail when made self-comparing; both τ and λ have mutation
tests that fail when the applying line is disabled.

All six self-reported statuses — **F1 PASS, F2 FAIL, F3 PASS, F5 PASS-with-caveat, F6 PASS, F7 FAIL** —
are **correct as stated**. This wave reports its own failures accurately, which is exactly what the x-wave
it replaces did not do.
