# Changelog

## 2.1.0 — Markov backbone for generation

v2.0 could score a token but could not order one. The n-gram statistics were
already being collected correctly — `P(σου | γεια) = 1.0` — but they entered
generation as three features among eighteen, feeding an expert whose weights
start at zero. A perfect ordering signal was diluted into noise.

### Fixed

* **Answers could not end.** `observe_sequence` recorded the teacher response
  without anchors, so `<eos>` was never a successor of anything (`count: 0`)
  and `<bos>` had no successors at all. The model had no representation of
  where a reply starts or stops, and generated until it hit the token cap,
  concatenating unrelated lessons. Sequences are now anchored as
  `<bos> … <eos>` via `observe_sequence(..., anchor=True)`.
* **The topic term punished silence.** The question/answer association score
  was a log-probability, so every token never seen with the question got a
  large negative score — including `<eos>`, which by construction never appears
  in an association table. It is now a bounded *bonus*, `log(1 + gain·p)`,
  which is exactly 0 without evidence.
* **`pos[1]`/`pos[2]` are skip-grams, not trigrams.** They answer "what appears
  two positions after X", not "what follows the pair (X, Y)". A real order-2
  context table was added; the skip-grams remain as features.

### Added

* `markov.py` — `MarkovScorer` with stupid backoff (Brants et al. 2007):
  trigram → bigram → unigram with α = 0.4, plus `perplexity()` for diagnostics.
* `TokenMemory.contexts` — bounded order-2 context table (real trigrams),
  capped by `max_contexts` and `max_successors_per_context`.
* `TokenMemory.associations` — bounded question-token → answer-token table.
  Without it a pure Markov chain starts every answer at `<bos>` and replies
  identically to every question.
* Generation is now a log-linear blend: `w_markov · log S + w_assoc · bonus +
  w_expert · correction`, with the expert contribution squashed to [-1, 1] so an
  unbounded logit cannot drown the language model.
* `tests/test_markov.py` — 38 tests covering anchoring, contexts, backoff,
  associations, candidate sets and end-to-end generation quality.
* Schema bumped to **v3**. v2 and v1 files still load; contexts and
  associations simply start empty and rebuild as the model learns.

### Changed

* `w_assoc` default 0.6 → **3.0** and `temperature` 0.7 → **0.1**, both chosen
  by measurement on two corpora (Greek and English), not by taste. On the Greek
  set the exact-reproduction rate went 55% → 93%; on the harder English set
  27% → 68%.
* `candidate_tokens` delegates to `MarkovScorer.candidates`, so generation and
  distillation always see the same candidates.
* `<eos>` is no longer offered at step 0 — an answer may end, but not be empty.
* `generate()` returns `finished`, indicating the model stopped on `<eos>`
  rather than hitting the cap.

### Measured effect

Teacher agreement over 120 benchmark turns went from 0.35 → **1.0**, and mean
reward from +0.55 → **+0.99**, with latency unchanged (20 ms/turn training,
11 ms inference) and 4.5 KB more state.

## 2.0.0 — teacher/student rewrite

The flat 372-line `spark_a2020a40.py` became the `spark_a2020a40/` package.
Nothing was deleted without a migration path: the old entry points still run and
the old memory file loads.

### Fixed (bugs in v1)

* **Greek was silently deleted.** `re.sub(r"[^a-z0-9_+#./:-]+", " ", text)`
  after `str.lower()` mapped every Greek character to whitespace, so the Greek
  keywords in `guess_tags` could never fire. Replaced with a
  `unicodedata.category`-driven scanner (NFC + `casefold`).
* **Silent total data loss.** `load_model` caught every exception with a bare
  `except:` and returned an empty model — one interrupted write erased
  everything. Loading is now validated, and falls back to `.bak` and then to the
  legacy file.
* **Non-atomic saves.** `save_model` wrote straight over the live file. Now:
  temp file → `flush` → `fsync` → `os.replace`, with a `.bak` kept.
* **Unbounded memory leak.** After the 110-item cap was hit, `add_token`
  returned `<UNK>` but `add_relation` still wrote `relations[target] += amount`
  for tokens that were never created, so relation dictionaries grew without
  limit. Relations are now capped per token.
* **The cap killed learning permanently.** `MAX_TOTAL_ITEMS = 110` with no
  eviction meant the model stopped learning after ~15 sentences. Replaced with
  configurable `id_bits` (default 12 = 4096 ids) and LRU eviction with
  frequency protection.
* **Silent id overflow.** `format(n, "07b")` produced 8-character ids past 127
  with no error.
* **Scale dominance.** `score = relation_weight + commonality` added a raw count
  to a raw weight, so the counter always won. All features are now normalised by
  an online Welford standardiser.
* **Hard-coded IP in three places** (`192.168.1.29`) and two different default
  models (`samantha-mistral` vs `tinyllama`). Now one `Config` with `SPARK_*`
  environment overrides and CLI flags.
* **`ollama_client.py` ran on import** — two `input()` calls and a `print` as
  import side effects, with an empty `IP_ADDRESS` that could never connect.
  Now an importable, side-effect-free wrapper.
* **120-second freeze when offline.** Every turn blocked on a 120 s timeout
  before printing a traceback. Availability is now probed with a 1.5 s timeout
  and cached for 30 s.
* **README pointed at `bit_tree_lm.py`**, a file that did not exist. It exists
  now (as a shim), and `python3 -m spark_a2020a40` is the documented entry.
* **Every token got identical tags**, making tags useless as a signal; and
  `guess_tags` produced `code`/`cyber`/`network`, which were not in
  `DEFAULT_TAGS` and were dropped once the cap was reached.
* **Unseeded `random.random()`** made generation untestable. All RNGs are now
  seeded from `Config.seed`.
* **Full JSON rewrite on every turn.** Autosave is now every 5 turns.
* **Dead code** removed (`if not ollama_host: ollama_host = IP_ADDRESS`).

### Added

* `AdaptiveQuadraticNeuron` — `z = Σw·f + Σq·f² + b`, AdaGrad steps, **proximal
  L1** on the curvature weights so unnecessary `q` reach exactly `0.0`.
* `ExpertPool` / `Expert` — usage, success, error/reward/confidence EMAs,
  specialisation signature, per-expert bounded replay, checkpoint + rollback,
  freeze, spawn/merge/prune with guards against runaway creation.
* `should_retrain(expert, context_stats)` — the single selective-retraining
  gate: sample count + cooldown + not frozen, then any of error / reward /
  confidence / teacher disagreement / drift / repeated class failure.
* `Router` — contextual bandit with an optimism bonus, Top-1 on
  `tiny_android` and Top-2 on `desktop_training`, usage distribution and
  entropy.
* `RewardEngine` — nine replaceable components, configurable weights, clipping,
  EMA. `relevance` is documented as a heuristic, not semantics.
* `FeatureExtractor` — the 18 specified token features plus 10 input features,
  with online normalisation so no feature dominates by scale.
* **Label-leakage guard** — `build_token_features()` has no `target_class_bit`
  parameter and raises `LabelLeakageError` for every label alias.
  `previous_predicted_class_bit` is the legal way to carry class context.
* `predict_input_class` / `train_class_predictor` — the binary class API.
* `TokenMemory` — order-1/2/3 positional statistics, per-token error EMA, class
  statistics, integer ids (binary strings only on export), bounded everywhere.
* `Persistence` — `schema_version`, atomic writes, validation, `.bak`
  recovery, compact mode, `migrate_v1`.
* `TeacherClient` / `OfflineTeacher` — cached availability probe, graceful
  degradation, no exceptions on the happy path.
* `TrainingCoordinator` — the twelve-step turn, distillation with
  positive/negative candidates, drift detection, lifecycle maintenance.
* Two profiles: `tiny_android` (default) and `desktop_training`.
* Full CLI: `:q :save :stats :student :teacher :feedback :experts :expert
  :freeze :unfreeze :train :retrain :memory :class :debug :help`.
* `benchmark.py` — latency, peak memory, active experts, reward/accuracy,
  memory JSON size.
* **11 test modules, 261 tests**, standard-library `unittest` only, including
  the mandated adaptive-neuron regressions (circle, linear, the six-point
  dataset, L1-versus-necessary-curvature) and the selective-retraining test.
* `examples/migrate_legacy_memory.py` and `examples/session_example.md`.

### Changed

* `spark_a2020a40.py` is now a shim that forwards to the package. The old
  helper names (`simple_tokenize`, `load_model`, `generate_student_text`,
  `interactive`) still exist and are marked deprecated.
* `ollama_client.generate_text` keeps its name and gained `host` / `port` /
  `timeout` parameters.
* Memory file default is `spark_memory.json`; `bittreelm_memory.json` is read
  and migrated but never written to.

### Compatibility notes

* The runtime avoids f-strings and `dataclasses` for old Android interpreters.
* `python3 spark_a2020a40.py`, `python3 bit_tree_lm.py` and
  `python3 -m spark_a2020a40` are all valid entry points.

## 1.0.0 — original BitTreeLM

Single-file symbolic model: binary token ids, tag dictionary, weighted relation
counts, decision-tree-style generation, Ollama teacher, 110-item cap.
