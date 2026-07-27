# SparkA2020A40

**An experimental teacher–student chatbot small enough to run on a 2015 Android
phone.** No PyTorch, no TensorFlow, no JAX, no scikit-learn, no NumPy — the
student runtime is pure Python standard library.

The project is named after the **Lenovo A2020a40** (Android 5.1.1, 1 GB RAM,
8 GB storage), the long-term compatibility target.

> v2.0.0 is a full rewrite of the internals. The old flat `spark_a2020a40.py`
> still works as an entry point and the old `bittreelm_memory.json` migrates
> automatically. See [CHANGELOG.md](CHANGELOG.md).

---

## The idea

Not one big neural network — **many tiny adaptive ones**, plus a router that
consults only the right one or two per context.

```
              ┌──────────────┐
   input ───► │ FeatureExtr. │──► input features (10)
              └──────┬───────┘
                     ├──────────────► class predictor ──► input_class_bit
                     ▼
              ┌──────────────┐
              │    Router    │  contextual bandit, LinUCB-flavoured
              └──────┬───────┘
                     │ Top-1 (tiny_android) / Top-2 (desktop)
                     ▼
        ┌────────────────────────────┐
        │  ExpertPool (max 6 / 24)   │  each expert:
        │  z = Σw·f + Σq·f² + b      │  2n+1 parameters
        └────────────┬───────────────┘
                     ▼
              ┌──────────────┐        ┌──────────────┐
              │ TokenMemory  │◄──────►│ RewardEngine │
              │ bounded LRU  │        │  clipped+EMA │
              └──────────────┘        └──────────────┘
```

The **teacher** is an Ollama model on a laptop. The **student** is everything
above and keeps working when the laptop is off.

### The adaptive quadratic scorer

Each expert scores a candidate token with

```
z = Σ(wᵢ·fᵢ) + Σ(qᵢ·fᵢ²) + b        p = sigmoid(z)
```

The curvature weights `q` carry an **L1 penalty applied as a proximal
soft-thresholding step**, not as a subgradient. That distinction is the whole
point: subgradient L1 leaves weights hovering at `1e-9`, while the proximal
operator drives them to *exactly* `0.0`. So a linearly separable problem
provably ends up linear, and only genuinely non-linear problems keep curvature.

`tests/test_adaptive_neuron.py` asserts this on the dataset from the spec:

| dataset | accuracy | max &#124;q&#124; | boundary |
|---|---|---|---|
| `[[1,2,0],[2,3,0],[3,4,0],[4,3,1],[3,2,1],[2,1,1]]` | 100 % | **exactly 0.0** | `x − y = 0` |
| points inside/outside a circle | 99 % | ≈ 1.0 (both negative) | circular |

---

## Installation

```bash
git clone https://github.com/AlexandrosGiann/SparkA2020A40.git
cd SparkA2020A40
python3 -m spark_a2020a40 --offline
```

There is nothing to `pip install`. Python 3.4+ is enough for the runtime;
the test suite uses `unittest` from the standard library.

Run the tests:

```bash
python3 -m unittest discover -s tests -t . -v
```

---

## Ollama setup (teacher mode)

On the laptop:

```bash
ollama pull tinyllama          # or samantha-mistral, llama3.2, ...
```

Ollama listens on localhost only by default. To reach it from the phone:

```ini
# /etc/systemd/system/ollama.service.d/override.conf
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
```

```bash
sudo systemctl daemon-reload && sudo systemctl restart ollama
ss -ltnp | grep 11434          # expect *:11434
```

Check from the phone's browser: `http://YOUR_PC_IP:11434` should say
`Ollama is running`.

Then, on the phone:

```bash
python3 -m spark_a2020a40 --host 192.168.1.29 --model tinyllama
```

The host is no longer hard-coded anywhere. Every setting can also come from the
environment:

```bash
export SPARK_OLLAMA_HOST=192.168.1.29
export SPARK_OLLAMA_MODEL=tinyllama
export SPARK_PROFILE=tiny_android
```

---

## Offline student mode

```bash
python3 -m spark_a2020a40 --offline
python3 -m spark_a2020a40 --offline --ask "γεια σου"
```

When the teacher is unreachable the client **does not block and does not
crash**. Availability is probed with a 1.5 s timeout and the result is cached
for 30 s; the turn simply continues without a teacher, computes a reward, and
updates the routed experts as usual.

---

## Training

A turn runs twelve steps:

1. receive input
2. extract input features
3. predict `input_class_bit`
4. ask the Ollama teacher *(skipped when offline)*
5. receive the teacher response
6. tokenize it
7. build positive and negative candidate tokens
8. supervised distillation update
9. student generation
10. compute the reward
11. update **only the routed experts**
12. save compact memory

Positive examples are the teacher's actual next tokens; negatives are drawn
from the student's own candidate set, so the student learns to rank its own
mistakes below the teacher's choices.

### Selective retraining

Training all experts on every token would defeat the architecture. An expert is
retrained only when **all** of

* its replay buffer holds enough samples,
* its cooldown has elapsed,
* it is not frozen,

and **at least one** of

* `error_ema` above threshold,
* `reward_ema` below threshold,
* `confidence_ema` below threshold,
* significant teacher/student disagreement,
* detected distribution drift,
* repeated failure on one class.

Retraining mixes recent and older replay samples, holds out a deterministic
validation subset, and **rolls back** to the previous checkpoint if validation
error gets worse. Stable, good experts are frozen and stop being updated.

`:retrain` printing *"no expert met the retraining conditions"* is the system
working correctly, not a failure.

### Expert lifecycle

| action | conditions |
|---|---|
| **spawn** | every expert has low confidence **and** enough similar failures accumulated **and** `max_experts` not reached |
| **merge** | similar weights **and** similar contexts **and** the merge does not hurt validation (otherwise it is rolled back) |
| **prune** | barely used **and** chronically low reward **and** another expert covers the same region **and** a checkpoint is stored |
| **freeze** | untouched for `freeze_stability_steps` with low error and good reward |

Spawn pressure decays on every success, so a bad patch cannot trigger runaway
expert creation.

---

## Reinforcement feedback

The reward is a weighted, clipped, EMA-smoothed sum of replaceable components:

```
reward = w_teacher·teacher_agreement + w_class·class_correctness
       + w_user·user_feedback        + w_relevance·relevance
       + w_completion·sequence_completion
       - w_repetition·repetition     - w_invalid·invalid_output
       - w_length·excessive_length   - w_uncertainty·uncertainty
```

From the CLI:

```
You: :feedback +1
applied reward +1.000
```

The reward reaches only the experts that were actually routed on that turn, and
their router arms.

> **`relevance` is a bag-of-tokens overlap heuristic, not semantic
> understanding.** It is registered like every other component precisely so it
> can be replaced:
>
> ```python
> engine.register_component("relevance", my_better_scorer, weight=0.6)
> ```

Every weight is tunable via `SPARK_W_TEACHER`, `SPARK_W_USER`, and so on.

---

## Expert management (CLI)

```
:q                quit and save            :save          save now
:stats            full statistics          :memory        memory summary
:student <text>   answer offline only      :teacher <txt> ask Ollama only
:feedback +1|-1   reinforce last answer    :class <text>  predict class bit
:experts          list experts             :expert <id>   expert detail
:freeze <id>      freeze an expert         :unfreeze <id> unfreeze it
:train            one supervised pass      :retrain       selective retrain
:debug on|off     verbose turn reports     :help          this text
```

`:stats` reports token count, relation count, expert count, active/frozen
split, average reward, average error, memory file size and the router usage
distribution.

---

## Memory and storage

* **Bounded.** Token count is capped by `max_tokens` with **LRU eviction plus
  frequency protection** — the model never stops learning, unlike the old hard
  wall at 110 items. Relation tables are capped per token. Replay buffers are
  ring buffers. There is no unbounded cache anywhere.
* **Versioned.** Every payload carries `schema_version`; a newer schema is
  refused rather than misread.
* **Atomic.** Write to a temp file → `flush` → `fsync` → `os.replace`.
* **Recoverable.** The previous good file is kept as `.bak` and used
  automatically when the primary file is truncated or corrupt.
* **Compact mode.** `tiny_android` writes minified JSON with rounded floats.

### Migrating the old memory

Automatic on first run, or explicitly:

```bash
python3 examples/migrate_legacy_memory.py bittreelm_memory.json spark_memory.json
```

```
legacy file : bittreelm_memory.json
  tokens    : 9
  tags      : 7
migrated to schema v2
  tokens    : 11
  relations : 13
  legacy tags preserved under state['legacy']['tags']: ['ai', 'general', ...]
```

The 7-bit binary ids are decoded back to integers and preserved,
`commonality` becomes the token count, `relations` become order-1 positional
statistics, and the legacy `tags` are kept under `state['legacy']['tags']` so
nothing is lost. The old file is never modified.

---

## Tokenizer

Greek and English, driven by `unicodedata.category` rather than an ASCII
blacklist:

```python
>>> Tokenizer().tokenize("Καλημέρα, ΚΟΣΜΕ! Πώς είσαι;")
['καλημέρα', ',', 'κοσμε', '!', 'πώσ', 'είσαι', ';']
>>> Tokenizer().tokenize("def f(x): return x**2 != 3")
['def', 'f', '(', 'x', ')', ':', 'return', 'x', '**', '2', '!=', '3']
>>> Tokenizer().tokenize("δες https://example.com/a?b=1 τώρα")
['δες', 'https://example.com/a?b=1', 'τώρα']
```

NFC normalisation plus `casefold()`, which also folds Greek final sigma
(`λόγος` → `λόγοσ`) so both spellings hit the same memory key. Accents are
preserved. Numbers, URLs, multi-character operators and identifiers survive.
Over-long tokens are split, not dropped.

---

## Features

For each candidate next token (18 values, all normalised to a common scale by
an online Welford standardiser):

| # | feature |
|---|---|
| 1 | `commonality_in_data` — frequency ÷ max frequency |
| 2–4 | `length_token`, `length_input`, `length_output` |
| 5–7 | `common_pos_1/2/3` — P(candidate \| token 1/2/3 positions back) |
| 8 | `input_class_bit` |
| 9 | `previous_predicted_class_bit` |
| 10 | `error_probability` — EMA of past errors, never future information |
| 11–13 | `repeats_in_input`, `repeats_in_output`, `repeats_in_window` |
| 14–18 | `ord_sum`, `ord_mean`, `ord_weighted_sum`, `…_mod_257`, `…_mod_263` |

`ord_sum` is kept exactly as specified (a plain sum over NFC + casefolded
text). Because a plain sum collides on anagrams — `ord_features("abc")[0] ==
ord_features("cba")[0]` — the position-weighted variants are provided
**alongside** it, never as a replacement.

### No label leakage

`build_token_features()` has **no `target_class_bit` parameter at all**.
Passing one raises `LabelLeakageError`:

```python
>>> extractor.build_token_features("x", target_class_bit=1)
LabelLeakageError: build_token_features() refuses label(s) target_class_bit as
input features; use previous_predicted_class_bit for the previous timestep instead
```

The previous timestep's *prediction* is a legitimate context feature and is
exposed separately. `tests/test_no_label_leakage.py` checks the signature, every
label alias, and asserts that no single feature column perfectly separates
positives from negatives in the distillation batch.

---

## Profiles

```python
profile = "tiny_android"     # the default and the priority
profile = "desktop_training" # roomier limits for distillation runs
```

| | tiny_android | desktop_training |
|---|---|---|
| `max_experts` | 6 | 24 |
| `max_tokens` | 1 500 | 20 000 |
| `router_top_k` | 1 | 2 |
| relations/token | 12 | 48 |
| replay/expert | 48 | 512 |
| compact JSON | yes | no |

---

## Benchmark

```bash
python3 benchmark.py --turns 120 --both
```

Measured on x86-64 Python 3.10 (a 2015 phone will be roughly 10–20× slower):

| metric | tiny_android | desktop_training |
|---|---|---|
| training turn (mean / p95) | 22.1 / 41.5 ms | 101.1 / 223.1 ms |
| inference only (mean / p95) | 9.5 / 19.6 ms | 83.2 / 219.3 ms |
| peak RSS | 20.7 MB | 24.1 MB |
| active experts | 1 / 6 | 1 / 24 |
| class accuracy | 0.975 | 0.975 |
| teacher agreement (1st → 2nd half) | 0.345 → 0.408 | 0.407 → 0.382 |
| memory JSON | 9.0 KB | 115.5 KB |

Peak memory comes from `resource.getrusage` on Linux/Android and falls back to
`tracemalloc` elsewhere; the report always labels the source.

---

## Before you run it on a phone

```bash
python3 check_device.py                      # offline checks
python3 check_device.py --host 192.168.1.29  # also probe the teacher
```

`check_device.py` verifies the interpreter version, the required builtins
(`os.replace`, `str.casefold`, `deque(maxlen=)`, `Request(method=)`), Unicode
handling, atomic writes on that filesystem, free storage, and then times twelve
real training and inference turns and reports peak RSS. It is written to parse
on very old Python 3 and checks the interpreter *before* importing anything from
the package, so it gives a useful answer even where the package cannot run.

## Android limitations

* **Interpreter age.** QPython 3H on Android 5.1.1 ships an old Python 3. The
  runtime modules therefore avoid f-strings and `dataclasses` and use
  `str.format()` throughout. The one modern convenience — the lazy
  `__getattr__` in `__init__.py` (PEP 562, Python 3.7+) — degrades gracefully:
  submodule imports keep working on older interpreters.
* **`os.replace` needs Python 3.3+.** It is atomic on Android's ext4/f2fs.
* **RAM.** ~20 MB peak on desktop, most of which is the interpreter itself.
  Stay on `tiny_android`; raising `max_tokens` or `max_experts` is the fastest
  way to get killed by the low-memory reaper on a 1 GB device.
* **Flash wear.** Autosave is every 5 turns (`SPARK_AUTOSAVE_EVERY`), not every
  turn as in v1. Each save rewrites the whole JSON.
* **No threads, no async.** One process, one loop.
* **Network.** The teacher probe has a 1.5 s timeout and is cached for 30 s, so
  a phone off the LAN is not punished on every turn.
* **Pydroid 3** (tested on a Redmi Note 11 / Android 12) is the easier path;
  QPython 3H on the A2020a40 is the harder target and is still unverified.

---

## Repository layout

```
spark_a2020a40/
    __init__.py        version + lazy exports
    config.py          profiles, env overrides, validation
    tokenizer.py       Unicode Greek/Latin/code tokenizer
    features.py        18 token features + 10 input features, normalisation
    memory.py          bounded LRU symbolic memory, order-1/2/3 statistics
    adaptive_neuron.py z = Σw·f + Σq·f² + b, proximal-L1 curvature
    experts.py         Expert, ExpertPool, should_retrain
    router.py          contextual bandit, Top-k selection
    rewards.py         pluggable reward components
    replay.py          bounded ring buffers + deterministic validation split
    teacher.py         Ollama client with offline degradation
    student.py         routing, scoring, generation, class head
    trainer.py         the twelve-step turn
    persistence.py     versioned + atomic + recoverable + v1 migration
    cli.py             the REPL
tests/                 10 test modules, standard-library unittest
examples/              migration script, annotated session
benchmark.py           latency / memory / experts / reward / JSON size
check_device.py        device readiness check -- run this first on a new phone
spark_a2020a40.py      legacy entry point (shim)
bit_tree_lm.py         legacy entry point (shim, as named in the v1 README)
ollama_client.py       deprecated wrapper over teacher.py
```

---

## Status

| | |
|---|---|
| Greek + English tokenizer | working |
| student offline | working |
| teacher via Ollama | working |
| selective retraining + rollback | working |
| expert spawn/merge/prune/freeze | working |
| versioned atomic persistence | working |
| v1 memory migration | working |
| Lenovo A2020a40 | **not yet verified** |

## Disclaimer

Experimental and educational. The model produces short, often nonsensical
output — it is a compression and constrained-device experiment, not a
replacement for a real LLM.

## Author

Alexandros Giannakis — <https://github.com/AlexandrosGiann>
