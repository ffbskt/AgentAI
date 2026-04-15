# Task: RLVR curriculum for a tiny arithmetic transformer

## Goal

Upgrade the current tiny transformer (about 100K parameters) from **final-answer addition** to **longer symbolic reasoning chains**, then apply **RLVR** so the model prefers valid reasoning traces instead of shortcutting directly to the answer.

The system must stay in a **small-symbol vocabulary regime**. No natural-language explanations in samples. Reasoning traces must use only arithmetic symbols and digits.

The attached DeepSeekMath paper motivates the overall sequence:

1. strong base competence first,
2. supervised reasoning traces next,
3. reinforcement learning after that,
4. reward should be verifiable,
5. process-level signals are useful, not only final-answer signals.

DeepSeekMath specifically uses:
- supervised fine-tuning with chain-of-thought / program-of-thought style reasoning,
- then RL with GRPO,
- and distinguishes outcome supervision from process supervision.

For this project, adapt those ideas to a tiny arithmetic model with a rigid symbolic format.

---

## Core design decisions

### 1) No natural language in data
All training and evaluation samples must be symbolic only.

### 2) Final answer marker
Do **not** introduce verbose tags if avoidable.

Use this rule:

- every rewrite step is separated by `=`
- the **final answer is always the substring after the last `=`**

Example:

`123+458=100+20+3+400+50+8=500+70+11=581`

Here:
- original problem: `123+458`
- intermediate steps: `100+20+3+400+50+8`, `500+70+11`
- final answer: `581`

This keeps the format compatible with a tiny vocab.

### 3) Rigid symbolic trace format
Do not allow free-form styles initially.
Start with one canonical style per phase.

### 4) Curriculum first, RLVR second
RLVR should not start from a weak random policy.
Each phase must begin with short supervised adaptation and only then try RLVR.

### 5) Verifiable reward
Reward must be computed from deterministic arithmetic checks, not from another LLM.

---

## What the agent must build

Build a **comfortable and modular experiment framework** so the learning sequence can be changed easily.

Required modules:

1. `format.py`
   - defines symbolic sample formats for each phase
   - converts numbers/problems into canonical traces
   - extracts final answer as the suffix after the last `=`

2. `dataset.py`
   - procedural generation of train / validation / test sets
   - configurable by difficulty phase
   - supports both final-answer-only and reasoning-trace targets

3. `verify.py`
   - parses symbolic trajectories
   - checks syntax validity
   - checks local step equivalence
   - checks final correctness
   - returns dense reward components

4. `curriculum.py`
   - defines ordered learning phases
   - allows easy reordering / skipping / changing phases
   - holds configs for data format, max digits, carries, epochs, RL on/off

5. `train_sft.py`
   - runs short supervised fine-tuning for a given phase

6. `train_rlvr.py`
   - runs RLVR for a given phase
   - samples several trajectories per problem
   - computes rewards with `verify.py`
   - updates the model relative to the sampled group

7. `evaluate.py`
   - runs evaluation after each phase
   - logs final-answer accuracy
   - logs trace validity
   - logs step validity
   - logs exact-trace success where appropriate
   - logs average generated length

8. `run_phase.py`
   - given a phase config, perform:
     1. short SFT warmup,
     2. short test evaluation,
     3. short RLVR run,
     4. test evaluation again,
     5. save metrics and examples

9. `run_curriculum.py`
   - executes all phases in order
   - supports resume from checkpoint
   - writes summary report

10. `TASK_RESULTS.md` generator
   - after experiments, automatically write a concise markdown report:
     - phase settings
     - before/after metrics
     - whether the phase worked
     - example successful traces
     - example failures
     - recommendation for the next phase

---

## Symbolic formats to support

The framework must support several symbolic target styles, because later phases will get harder.

### Format A: final answer only
Example:
`7+8=15`

Use only as a baseline or warmup.

### Format B: full decomposition
Example:
`22+9=20+2+9=20+11=31`

### Format C: two-number decomposition
Example:
`47+18=40+7+10+8=50+15=65`

### Format D: carry-oriented compressed symbolic format
Example:
`58+67=15|12|1=125`

Only add this **later** if the earlier formats work.
The exact compressed carry notation may be changed, but it must be:
- purely symbolic,
- deterministic,
- easy to verify.

### Format E: mixed curriculum mode
A configurable mixture of:
- final-answer-only,
- short traces,
- longer traces.

Do not use mixed mode at the beginning.

---

## Recommended curriculum

The agent must implement the curriculum as editable config, not hardcoded logic.

Default recommended sequence:

### Phase 0 — Baseline check
Goal:
- verify the current model really solves its existing task.

Data:
- same current task the model already knows (3-digit sum prediction if that is the current capability)

Train:
- no major retraining yet
- only evaluation and optional tiny refresh SFT

Success criteria:
- stable baseline on held-out data
- generation syntax already controlled

---

### Phase 1 — 1-digit + 1-digit, final answer only
Goal:
- establish stable controlled training / eval pipeline.

Format:
- Format A

Examples:
- `2+7=9`
- `8+5=13`

Train:
- short SFT only

RLVR:
- optional
- if used, only final-answer reward

Success criteria:
- very high held-out exact accuracy
- no syntax failures

---

### Phase 2 — 2-digit + 1-digit, short decomposition traces
Goal:
- teach first explicit thought chain.

Format:
- Format B

Examples:
- `22+9=20+2+9=20+11=31`
- `34+5=30+4+5=30+9=39`

Train:
- short SFT warmup
- then RLVR

Reward:
- syntax reward
- local-step equivalence reward
- final-answer reward

Success criteria:
- model produces parseable traces
- step-valid traces increase after RLVR
- answer accuracy does not collapse

---

### Phase 3 — 2-digit + 2-digit without carry explosion
Goal:
- learn longer traces while keeping difficulty moderate.

Format:
- Format C

Examples:
- `21+13=20+1+10+3=30+4=34`
- `42+15=40+2+10+5=50+7=57`

Data restriction:
- initially bias toward cases with simple carries or no carry

Train:
- SFT then RLVR

Success criteria:
- trace validity survives longer sequences
- RLVR improves process correctness, not just final answer

---

### Phase 4 — 2-digit + 2-digit with carries
Goal:
- make the model handle nontrivial carry behavior.

Format:
- Format C

Examples:
- `47+18=40+7+10+8=50+15=65`
- `58+27=50+8+20+7=70+15=85`

Train:
- SFT warmup
- RLVR with stronger process reward

Success criteria:
- improvement on carry cases specifically
- fewer malformed rewrites

---

### Phase 5 — 3-digit + 2-digit and 3-digit + 3-digit
Goal:
- extend chain length and keep verification stable.

Format:
- Format C or a carefully designed longer decomposition variant

Examples:
- `123+45=100+20+3+40+5=160+8=168`
- `123+458=100+20+3+400+50+8=500+70+11=581`

Train:
- SFT warmup
- RLVR

Success criteria:
- no sharp drop in syntax validity
- answer accuracy and trace validity both improve

---

### Phase 6 — compressed carry notation (optional)
Goal:
- test whether a shorter internal reasoning code works better than long decompositions.

Use only if previous phases already work.

Train:
- start with SFT on deterministic compressed notation
- then RLVR

Success criteria:
- shorter traces with preserved correctness
- compare against long decomposition traces

---

### Phase 7 — mixed training / robustness
Goal:
- prevent the model from overfitting to one exact length or one exact formatting pattern.

Data:
- mixture of earlier successful formats
- controlled proportions from config

Train:
- short SFT
- RLVR with process reward retained

Success criteria:
- robust performance across multiple held-out formats
- no severe collapse to answer-only shortcutting

---

## RLVR design

Start with the smallest stable RLVR loop.

### Sampling
For each problem:
- sample `N` candidate trajectories from the current policy
- recommended starting values: `N = 4` or `N = 8`

Do **not** start with very large group sizes.

### Reward components
Use dense and interpretable reward.

Recommended default:

- `r_syntax`
  - positive if the output can be parsed
  - negative if malformed

- `r_steps`
  - reward proportional to the fraction of locally valid rewrite steps

- `r_final`
  - positive if final answer is correct
  - zero or negative if incorrect

- `r_length`
  - mild penalty for unnecessary extra tokens

Total reward:
`R = w_syntax * r_syntax + w_steps * r_steps + w_final * r_final - w_length * r_length`

Expose all weights in config.

### Why process reward matters
Do not rely only on final correctness.
A tiny model may learn to emit garbage traces and still occasionally hit the final answer.
Rewarding local step validity helps push the model toward real symbolic reasoning rather than answer-only shortcutting.

### Relative update
Use a **group-relative** style update:
- compare candidates sampled for the same prompt
- push up outputs above the group average
- push down outputs below the group average

This is closer to the GRPO idea and avoids unnecessary extra complexity for a tiny model.

### KL control
Retain a reference-policy anchor or equivalent regularization.
Without this, the model may drift and forget its base competence.

### Fallback option
Also implement a simpler baseline:
- Best-of-N / rejection-sampling fine-tuning

That means:
1. sample multiple trajectories,
2. keep the best verified ones,
3. fine-tune on those.

This baseline is often easier and should be compared against RLVR.

---

## Verifier requirements

The verifier is critical. It must be deterministic and well tested.

It must expose at least:

### `parse_trace(s)`
Input:
- generated symbolic string

Output:
- parsed list of expressions split by `=`
- parse status
- error code if malformed

### `eval_expr(expr)`
Evaluate a `+`-only expression like:
- `20+2+9`
- `500+70+11`

### `check_local_steps(parts)`
For each adjacent pair:
- evaluate left side
- evaluate right side
- confirm equality

For:
`22+9=20+2+9=20+11=31`
checks:
- `22+9 == 20+2+9`
- `20+2+9 == 20+11`
- `20+11 == 31`

### `extract_final_answer(parts)`
Return the substring after the last `=`

### `score_trace(problem, trace)`
Return structured scores:
- parse_ok
- final_ok
- num_steps
- valid_step_fraction
- reward_total
- error_type

### Unit tests
Write explicit tests for:
- valid traces
- wrong final answer
- wrong middle step
- malformed token order
- empty segment
- doubled operators
- leading/trailing separators
- adversarial weird but parseable cases

---

## Dataset generation requirements

The dataset generator must support configurable difficulty knobs:

- max digits of each addend
- exact shape:
  - `1d+1d`
  - `2d+1d`
  - `2d+2d`
  - `3d+2d`
  - `3d+3d`
- carry frequency:
  - no carry
  - easy carry
  - heavy carry
- target format:
  - A / B / C / D / mixed
- number of samples
- seed
- train / val / test split

Important:
- test sets must be generated with separate seeds
- include dedicated edge-case test suites:
  - `9+9`
  - `19+9`
  - `99+1`
  - `58+67`
  - `199+801`
  - long carry chains

---

## What “working” means in each phase

The agent must not just run training.
It must decide whether the phase worked.

For every phase, record at least:

1. exact final-answer accuracy
2. parseable-trace rate
3. full local-step-valid rate
4. average valid-step fraction
5. exact full-trace correctness rate
6. average output length
7. train reward stats if RLVR was used
8. comparison before vs after RLVR

A phase counts as **worked** only if:
- final-answer accuracy is not worse than baseline by more than a small tolerance,
- and at least one process metric improves materially.

Recommended default rule:
- worked if final-answer accuracy does not drop by more than 1 absolute point
- and either parseable-trace rate or full-step-valid rate improves by at least 3 absolute points

Make this threshold configurable.

---

## Minimal experiment protocol per phase

Each phase must run the same compact protocol first.

### Step 1 — SFT warmup
- train only a few short epochs
- verify loss decreases
- evaluate on validation set

### Step 2 — generation sanity check
Generate examples from held-out problems and inspect:
- format adherence
- whether the final answer is after the last `=`
- whether steps are mostly valid

### Step 3 — RLVR short run
- run a short RLVR experiment
- log reward statistics
- evaluate again

### Step 4 — decision
Mark the phase as:
- worked
- unclear
- failed

### Step 5 — save examples
For each phase save:
- 10 successful examples
- 10 failure examples
- 10 examples before RLVR and after RLVR on the same prompts

---

## Search / literature tasks for the agent

Before implementing the final RL loop, the agent must briefly check and summarize the most relevant guidance from:

1. the attached DeepSeekMath paper
2. a small set of current web sources / docs focused on RLVR and GRPO practice

The summary should stay concise and extract only actionable advice for this tiny project.

Expected takeaways to verify against implementation:
- reasoning training should come after base competence
- process-level signals are useful
- group-relative RL is a sensible fit
- KL regularization matters
- early short experiments are important before scaling up

Add the summary to the top of `TASK_RESULTS.md`.

---

## Strong implementation preferences

1. Keep configuration centralized.
2. Keep every phase reproducible by seed.
3. Avoid hidden magic constants.
4. Make reward weights easy to change.
5. Make it easy to switch:
   - SFT only
   - best-of-N fine-tuning
   - RLVR
6. Save checkpoints after every phase.
7. Save all evaluation tables as markdown and CSV.
8. Keep the whole system small and inspectable.

---

## Explicit comparison baselines

The agent must compare at least these three modes on one or more nontrivial phases:

### Baseline A
Final-answer SFT only

### Baseline B
Reasoning-trace SFT only

### Baseline C
Reasoning-trace SFT + RLVR

Optional:
### Baseline D
Reasoning-trace SFT + best-of-N self-training

This comparison matters. It is possible that for a tiny model, simple self-training gives most of the gain.

---

## Important failure checks

The agent must actively check for:

1. **answer-only shortcutting**
   - final answer correct but steps mostly invalid

2. **trace babbling**
   - long outputs with bad syntax

3. **reward hacking**
   - patterns that exploit the verifier without valid reasoning

4. **policy drift**
   - RLVR hurts previously learned competence

5. **overfitting to one exact trace length**
   - good on training-style traces, weak on nearby variants

---

## Concrete deliverables

1. `task_rlvr_tiny_transformer.md`
   - this task file, refined if needed

2. code modules listed above

3. config files for all phases

4. a short experiment run for every phase
   - only a few epochs / short RLVR budget at first

5. `TASK_RESULTS.md`
   - concise report with metrics and decision for each phase

6. recommendation:
   - which phase sequence seems to work best
   - whether RLVR truly improves reasoning traces over SFT alone

---

## Final instruction to the agent

Implement the framework first, then run **small validation experiments phase by phase**.

Do not jump straight to a large final training run.

For each phase:
1. run a short SFT warmup,
2. evaluate,
3. run a short RLVR test,
4. evaluate again,
5. decide whether the phase worked,
6. only then continue to the next phase.

The objective is not only higher final-answer accuracy.
The objective is to make the model produce **longer, valid, verifiable symbolic thought chains** and then use RLVR so those thought chains become more likely than answer-only shortcuts.
