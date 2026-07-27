# -*- coding: utf-8 -*-
"""Benchmark the student runtime.

Measures response time, peak memory (where the platform exposes it), the number
of active experts, reward/accuracy and the size of the memory JSON.

    python3 benchmark.py --profile tiny_android --turns 200

Peak RSS comes from ``resource.getrusage`` on Linux/Android and falls back to
``tracemalloc`` (Python heap only) everywhere else, so the number is always
labelled with its source.
"""

import argparse
import gc
import json
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from spark_a2020a40.config import Config, PROFILES, PROFILE_TINY  # noqa: E402
from spark_a2020a40.persistence import Persistence  # noqa: E402
from spark_a2020a40.student import StudentModel  # noqa: E402
from spark_a2020a40.teacher import OfflineTeacher  # noqa: E402
from spark_a2020a40.trainer import TrainingCoordinator  # noqa: E402

LESSONS = [
    ("γεια σου", "γεια σου κόσμε πώς είσαι σήμερα φίλε"),
    ("τι κάνεις;", "καλά είμαι ευχαριστώ εσύ τι κάνεις σήμερα"),
    ("hello", "hello there how can i help you today"),
    ("what is python?", "python is a programming language used for scripts"),
    ("πες μου κάτι", "το μοντέλο μαθαίνει από τον δάσκαλο και απαντά μόνο του"),
    ("how does it work?", "the router picks one expert and only that one learns"),
]


def peak_memory():
    """Return ``(kilobytes, source)``."""
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports kilobytes, macOS reports bytes.
        if sys.platform == "darwin":
            usage = usage / 1024.0
        return float(usage), "rss"
    except (ImportError, AttributeError):
        pass
    try:
        import tracemalloc
        if tracemalloc.is_tracing():
            _, peak = tracemalloc.get_traced_memory()
            return peak / 1024.0, "python-heap"
    except ImportError:
        pass
    return 0.0, "unavailable"


def percentile(values, fraction):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def run_benchmark(profile=PROFILE_TINY, turns=120, workdir=None, verbose=True):
    cfg = Config(profile)
    temporary = workdir is None
    workdir = workdir or tempfile.mkdtemp(prefix="spark-bench-")
    path = os.path.join(workdir, "bench_memory.json")

    try:
        import tracemalloc
        tracemalloc.start()
    except ImportError:
        pass

    persistence = Persistence(cfg, path)
    student = StudentModel(cfg)
    coordinator = TrainingCoordinator(
        cfg, student, OfflineTeacher(cfg), persistence=persistence)

    gc.collect()
    train_latencies = []
    rewards = []
    agreements = []
    class_hits = 0

    start = time.time()
    for index in range(turns):
        prompt, answer = LESSONS[index % len(LESSONS)]
        turn_start = time.time()
        report = coordinator.process_turn(prompt, teacher_text=answer)
        train_latencies.append((time.time() - turn_start) * 1000.0)
        rewards.append(report["reward"])
        agreements.append(report["reward_breakdown"].get("teacher_agreement", 0.0))
        if report["class"]["predicted"] == report["class"]["target"]:
            class_hits += 1
    train_seconds = time.time() - start

    # Pure inference (the Android hot path).
    gc.collect()
    inference_latencies = []
    for index in range(max(20, turns // 4)):
        prompt = LESSONS[index % len(LESSONS)][0]
        inference_start = time.time()
        student.generate(prompt)
        inference_latencies.append((time.time() - inference_start) * 1000.0)

    coordinator.save()
    memory_bytes = persistence.size_bytes()
    kilobytes, source = peak_memory()
    pool_stats = student.pool.stats()

    half = max(1, len(rewards) // 2)
    result = {
        "profile": profile,
        "turns": turns,
        "train_total_seconds": round(train_seconds, 3),
        "train_ms_mean": round(sum(train_latencies) / len(train_latencies), 3),
        "train_ms_p95": round(percentile(train_latencies, 0.95), 3),
        "inference_ms_mean": round(
            sum(inference_latencies) / len(inference_latencies), 3),
        "inference_ms_p95": round(percentile(inference_latencies, 0.95), 3),
        "peak_memory_kb": round(kilobytes, 1),
        "peak_memory_source": source,
        "experts_total": pool_stats["count"],
        "experts_active": pool_stats["active"],
        "experts_frozen": pool_stats["frozen"],
        "expert_cap": cfg.max_experts,
        "tokens": len(student.memory),
        "token_cap": cfg.max_tokens,
        "relations": student.memory.total_relations(),
        "reward_mean": round(sum(rewards) / len(rewards), 4),
        "reward_mean_first_half": round(sum(rewards[:half]) / half, 4),
        "reward_mean_second_half": round(sum(rewards[half:]) / max(1, len(rewards) - half), 4),
        "teacher_agreement_first_half": round(sum(agreements[:half]) / half, 4),
        "teacher_agreement_second_half": round(
            sum(agreements[half:]) / max(1, len(agreements) - half), 4),
        "class_accuracy": round(class_hits / float(turns), 4),
        "memory_json_bytes": memory_bytes,
        "memory_json_kb": round(memory_bytes / 1024.0, 2),
        "router_entropy": round(student.router.entropy(), 4),
    }

    if temporary:
        shutil.rmtree(workdir, ignore_errors=True)

    if verbose:
        print_report(result)
    return result


def print_report(result):
    print("=" * 62)
    print("SparkA2020A40 benchmark -- profile: {0}".format(result["profile"]))
    print("=" * 62)
    rows = [
        ("turns", result["turns"]),
        ("training turn (mean / p95)", "{0} / {1} ms".format(
            result["train_ms_mean"], result["train_ms_p95"])),
        ("inference only (mean / p95)", "{0} / {1} ms".format(
            result["inference_ms_mean"], result["inference_ms_p95"])),
        ("peak memory ({0})".format(result["peak_memory_source"]),
         "{0} KB".format(result["peak_memory_kb"])),
        ("experts (active / frozen / cap)", "{0} / {1} / {2}".format(
            result["experts_active"], result["experts_frozen"], result["expert_cap"])),
        ("tokens / cap", "{0} / {1}".format(result["tokens"], result["token_cap"])),
        ("relations", result["relations"]),
        ("reward (first half -> second half)", "{0:+} -> {1:+}".format(
            result["reward_mean_first_half"], result["reward_mean_second_half"])),
        ("teacher agreement (first -> second)", "{0} -> {1}".format(
            result["teacher_agreement_first_half"],
            result["teacher_agreement_second_half"])),
        ("class accuracy", result["class_accuracy"]),
        ("memory JSON", "{0} KB".format(result["memory_json_kb"])),
        ("router entropy", result["router_entropy"]),
    ]
    for label, value in rows:
        print("{0:<38} {1}".format(label, value))
    print("=" * 62)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Benchmark the student runtime")
    parser.add_argument("--profile", choices=PROFILES, default=PROFILE_TINY)
    parser.add_argument("--turns", type=int, default=120)
    parser.add_argument("--both", action="store_true",
                        help="benchmark both profiles")
    parser.add_argument("--json", help="write the raw results to this file")
    args = parser.parse_args(argv)

    results = []
    profiles = PROFILES if args.both else (args.profile,)
    for profile in profiles:
        results.append(run_benchmark(profile, args.turns))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(results, handle, ensure_ascii=False, indent=2)
        print("wrote {0}".format(args.json))
    return 0


if __name__ == "__main__":
    sys.exit(main())
