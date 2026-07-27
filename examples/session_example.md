# Example teacher–student session

Everything below is real output from `python3 -m spark_a2020a40`, with the
legacy `bittreelm_memory.json` present in the working directory.

## 1. First start — the old memory migrates itself

```
$ python3 -m spark_a2020a40 --offline
SparkA2020A40 v2 -- tiny teacher/student chatbot
profile: tiny_android | tokens: 12 | experts: 1
teacher: offline (student only)
type :help for commands
```

The nine tokens and thirteen relations from `bittreelm_memory.json` were
decoded from their 7-bit binary ids and loaded as order-1 positional
statistics. The original file is left untouched.

## 2. A turn with the teacher online

```
$ python3 -m spark_a2020a40 --host 192.168.1.29 --model tinyllama --debug

You: γεια σου, τι κάνεις;

Teacher: Γεια σου! Είμαι καλά, ευχαριστώ. Εσύ τι κάνεις σήμερα;
Student: γεια σου είμαι καλά
  class: predicted=1 target=1 p=0.612
  experts: [0] | reward +0.734
  breakdown: class_correctness=+1.00, excessive_length=+0.00,
             invalid_output=+0.00, relevance=+0.50, repetition=+0.00,
             sequence_completion=+1.00, teacher_agreement=+0.44,
             uncertainty=+0.31, user_feedback=+0.00
  retrained: [] | lifecycle: {}
```

`retrained: []` is the expected outcome, not a bug: expert #0 is inside its
cooldown and none of its EMAs crossed a threshold, so nothing was retrained.
Only the single routed expert (`experts: [0]`) received the distillation
gradient.

## 3. Reinforcement from the user

```
You: :feedback +1
applied reward +1.000

You: :feedback -1
applied reward -1.000
```

The reward goes to the experts that were actually routed on the previous turn
and to their router arms — nowhere else.

## 4. The teacher disappears mid-session

Unplug the laptop and keep typing:

```
You: γεια σου ξανά

(teacher offline -- student only)
Student: γεια σου είμαι καλά ευχαριστώ
```

The probe fails once, is cached for 30 s, and the turn continues from step 9 of
the pipeline. No traceback, no 120-second freeze.

## 5. Inspecting the model

```
You: :stats
tokens .............. 60
relations ........... 134
experts ............. 1 (active 1, frozen 0)
average reward ...... +0.5467  (ema +0.5231)
average error ....... 0.2143
average confidence .. 0.5514
memory file ......... 9196 bytes
turns / steps ....... 120 / 120
router entropy ...... 0.000
router usage ........ #0:100%
teacher ............. offline @ http://192.168.1.29:11434/api/generate

You: :expert 0
expert #0 [active]
  usage 120  success 96 (80%)
  error_ema 0.2143  reward_ema +0.5231  confidence_ema 0.5514
  replay 48/48  steps_since_update 3  last_training_step 117
  curvature: 4 active of 18, max |q| = 0.31842
  bias +0.1204
```

`curvature: 4 active of 18` is the L1 penalty doing its job: fourteen of the
eighteen quadratic weights are exactly zero, so those features are being used
purely linearly.

## 6. Class prediction

```
You: :class Τι κάνεις σήμερα;
input_class_bit = 1  (p=0.8123, weak label=1)

You: :class Καλησπέρα.
input_class_bit = 0  (p=0.1904, weak label=0)
```

## 7. Selective retraining, on demand

```
You: :retrain
no expert met the retraining conditions (this is normal)

You: :train
trained 1 expert(s): #0:ok
```

`:retrain` respects `should_retrain`. `:train` bypasses the gate but still runs
validation and rolls back an update that makes the held-out set worse.

## 8. Quitting

```
You: :q
Saved. Bye.
```

The save is atomic: temp file → `fsync` → `os.replace`, with the previous file
kept as `spark_memory.json.bak`.
