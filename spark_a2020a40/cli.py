# -*- coding: utf-8 -*-
"""Interactive command line for the teacher/student loop."""

import argparse
import sys

from .config import Config, PROFILES, PROFILE_TINY, set_config
from .persistence import Persistence
from .rewards import RewardEngine
from .student import StudentModel
from .teacher import OfflineTeacher, TeacherClient
from .trainer import TrainingCoordinator

BANNER = "SparkA2020A40 v2 -- tiny teacher/student chatbot"

HELP_TEXT = """Commands
  :q                quit and save            :save          save now
  :stats            full statistics          :memory        memory summary
  :student <text>   answer offline only      :teacher <txt> ask Ollama only
  :teacher          teacher diagnostics
  :feedback +1|-1   reinforce last answer    :class <text>  predict class bit
  :experts          list experts             :expert <id>   expert detail
  :freeze <id>      freeze an expert         :unfreeze <id> unfreeze it
  :train            one supervised pass      :retrain       selective retrain
  :debug on|off     verbose turn reports     :help          this text
Anything else is a normal turn (teacher when reachable, student always)."""


def build_session(cfg, offline=False, memory_path=None):
    persistence = Persistence(cfg, memory_path)
    student = StudentModel(cfg)
    state = persistence.load()
    teacher = OfflineTeacher(cfg) if offline else TeacherClient(cfg)
    coordinator = TrainingCoordinator(
        cfg, student, teacher, RewardEngine(cfg), persistence)
    if state:
        coordinator.load_state(state)
    return coordinator, persistence


def format_stats(coordinator, persistence):
    stats = coordinator.stats()
    usage = stats["router_usage"]
    lines = [
        "tokens .............. {0}".format(stats["tokens"]),
        "relations ........... {0}".format(stats["relations"]),
        "experts ............. {0} (active {1}, frozen {2})".format(
            stats["experts"], stats["active_experts"], stats["frozen_experts"]),
        "average reward ...... {0:+.4f}  (ema {1:+.4f})".format(
            stats["avg_reward"], stats["reward_ema"]),
        "average error ....... {0:.4f}".format(stats["avg_error"]),
        "average confidence .. {0:.4f}".format(stats["avg_confidence"]),
        "memory file ......... {0} bytes".format(persistence.size_bytes()),
        "turns / steps ....... {0} / {1}".format(stats["turns"], stats["steps"]),
        "router entropy ...... {0:.3f}".format(stats["router_entropy"]),
        "router usage ........ " + (
            ", ".join("#{0}:{1:.0%}".format(k, v) for k, v in sorted(usage.items()))
            or "(none)"),
        "teacher ............. {0} @ {1}".format(
            "available" if stats["teacher"]["available"] else "offline",
            stats["teacher"]["url"]),
    ]
    return "\n".join(lines)


def format_expert(expert):
    neuron = expert.neuron
    return "\n".join([
        "expert #{0} [{1}]".format(expert.unique_id,
                                   "frozen" if expert.frozen else "active"),
        "  usage {0}  success {1} ({2:.0%})".format(
            expert.usage_count, expert.success_count, expert.success_rate()),
        "  error_ema {0:.4f}  reward_ema {1:+.4f}  confidence_ema {2:.4f}".format(
            expert.error_ema, expert.reward_ema, expert.confidence_ema),
        "  replay {0}/{1}  steps_since_update {2}  last_training_step {3}".format(
            len(expert.replay), expert.replay.capacity,
            expert.steps_since_update, expert.last_training_step),
        "  curvature: {0} active of {1}, max |q| = {2:.5f}".format(
            neuron.active_curvature(), neuron.n, neuron.curvature_magnitude()),
        "  bias {0:+.4f}".format(neuron.b),
    ])


def run(argv=None):
    parser = argparse.ArgumentParser(prog="spark", description=BANNER)
    parser.add_argument("--profile", choices=PROFILES, default=PROFILE_TINY)
    parser.add_argument("--host", help="Ollama host (overrides SPARK_OLLAMA_HOST)")
    parser.add_argument("--port", type=int, help="Ollama port")
    parser.add_argument("--model", help="Ollama teacher model name")
    parser.add_argument("--memory", help="path to the memory file")
    parser.add_argument("--offline", action="store_true",
                        help="never contact the teacher")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--ask", help="answer one prompt and exit")
    args = parser.parse_args(argv)

    cfg = Config(args.profile)
    if args.host:
        cfg.ollama_host = args.host
    if args.port:
        cfg.ollama_port = args.port
    if args.model:
        cfg.ollama_model = args.model
    if args.memory:
        cfg.memory_file = args.memory
    if args.debug:
        cfg.debug = True
    set_config(cfg)

    coordinator, persistence = build_session(cfg, args.offline, args.memory)

    if args.ask:
        print(coordinator.student.answer(args.ask))
        return 0

    print(BANNER)
    print("profile: {0} | tokens: {1} | experts: {2}".format(
        cfg.profile, len(coordinator.student.memory), len(coordinator.student.pool)))
    if args.offline:
        print("teacher: offline (student only)")
    elif coordinator.teacher.is_available():
        print("teacher: {0}".format(cfg.ollama_url()))
        ok, message = coordinator.teacher.check_model()
        if not ok:
            print("WARNING: {0}".format(message))
            print("         the server answers, but generation will fail -- "
                  "pass --model <name> or pull the model")
    else:
        print("teacher: unreachable at {0}".format(cfg.ollama_url()))
        print("         reason: {0}".format(
            coordinator.teacher.last_error or "no response"))
    print("type :help for commands")

    while True:
        try:
            line = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            coordinator.save()
            print("Saved. Bye.")
            return 0
        if not line:
            continue
        try:
            if not handle(line, coordinator, persistence, cfg):
                return 0
        except Exception as exc:  # keep the REPL alive
            print("error: {0}".format(exc))
    return 0


def handle(line, coordinator, persistence, cfg):
    """Return False to quit."""
    student = coordinator.student

    if line in (":q", ":quit", ":exit"):
        coordinator.save()
        print("Saved. Bye.")
        return False

    if line == ":help":
        print(HELP_TEXT)
        return True

    if line == ":save":
        print("saved" if coordinator.save() else
              "save failed: {0}".format(persistence.last_error))
        return True

    if line == ":stats":
        print(format_stats(coordinator, persistence))
        return True

    if line == ":memory":
        memory = student.memory
        top = sorted(memory.tokens.items(), key=lambda kv: -kv[1].count)[:12]
        print("tokens {0}/{1}, relations {2}, observations {3}".format(
            len(memory), cfg.max_tokens, memory.total_relations(),
            memory.total_observations))
        print("most common: " + ", ".join(
            "{0}({1})".format(text, record.count) for text, record in top))
        print("file: {0}".format(persistence.status()["path"]))
        return True

    if line == ":experts":
        for expert in sorted(student.pool, key=lambda e: e.unique_id):
            print("  #{0} {1} usage={2} err={3:.3f} rew={4:+.3f} conf={5:.3f} q_active={6}".format(
                expert.unique_id, "F" if expert.frozen else "A", expert.usage_count,
                expert.error_ema, expert.reward_ema, expert.confidence_ema,
                expert.neuron.active_curvature()))
        return True

    if line.startswith(":expert "):
        expert = student.pool.get(int(line.split(None, 1)[1]))
        print(format_expert(expert) if expert else "no such expert")
        return True

    if line.startswith(":freeze ") or line.startswith(":unfreeze "):
        command, argument = line.split(None, 1)
        expert = student.pool.get(int(argument))
        if expert is None:
            print("no such expert")
            return True
        expert.frozen = (command == ":freeze")
        print("expert #{0} is now {1}".format(
            expert.unique_id, "frozen" if expert.frozen else "active"))
        return True

    if line.startswith(":student "):
        print("Student: " + student.answer(line.split(None, 1)[1]))
        return True

    if line == ":teacher":
        status = coordinator.teacher.status()
        print("url ......... {0}".format(status["url"]))
        print("model ....... {0}".format(status["model"]))
        available = coordinator.teacher.is_available(force=True)
        print("reachable ... {0}".format("yes" if available else "no"))
        if available:
            names = coordinator.teacher.list_models(refresh=True)
            print("installed ... {0}".format(
                ", ".join(sorted(names)) if names else "(could not list)"))
            ok, message = coordinator.teacher.check_model()
            print("model check . {0}".format(message))
        if coordinator.teacher.last_error:
            print("last error .. {0}".format(coordinator.teacher.last_error))
        return True

    if line.startswith(":teacher "):
        prompt = line.split(None, 1)[1]
        if not coordinator.teacher.is_available():
            print("teacher unreachable: {0}".format(coordinator.teacher.last_error))
            return True
        answer = coordinator.teacher.generate(prompt)
        if answer is None:
            print("teacher failed: {0}".format(coordinator.teacher.last_error))
        else:
            print("Teacher: " + answer)
        return True

    if line.startswith(":feedback"):
        parts = line.split()
        value = 1 if (len(parts) < 2 or not parts[1].startswith("-")) else -1
        reward = coordinator.apply_feedback(value)
        print("no previous turn" if reward is None
              else "applied reward {0:+.3f}".format(reward))
        return True

    if line.startswith(":class "):
        text = line.split(None, 1)[1]
        probability, bit = student.classify_text(text)
        print("input_class_bit = {0}  (p={1:.4f}, weak label={2})".format(
            bit, probability, student.derive_target_class_bit(text)))
        return True

    if line == ":train":
        reports = coordinator.force_retrain(ignore_gate=True)
        print("trained {0} expert(s): {1}".format(
            len(reports), ", ".join(
                "#{0}:{1}".format(r["expert"], r["reason"]) for r in reports) or "-"))
        return True

    if line == ":retrain":
        reports = coordinator.force_retrain(ignore_gate=False)
        if not reports:
            print("no expert met the retraining conditions (this is normal)")
        else:
            for report in reports:
                print("  #{0} {1} applied={2} samples={3}".format(
                    report["expert"], report["reason"], report["applied"],
                    report["samples"]))
        return True

    if line.startswith(":debug"):
        cfg.debug = line.endswith("on")
        coordinator.debug = cfg.debug
        print("debug {0}".format("on" if cfg.debug else "off"))
        return True

    if line.startswith(":"):
        print("unknown command; try :help")
        return True

    report = coordinator.process_turn(line)
    if report["teacher_available"]:
        print("\nTeacher: " + (report["teacher_text"] or "").strip())
    else:
        reason = coordinator.teacher.last_error
        if reason:
            print("\n(teacher failed -- student only)\n  reason: {0}".format(reason))
        else:
            print("\n(teacher offline -- student only)")
    if report["student_text"]:
        print("Student: " + report["student_text"])
    elif len(student.memory) <= 8:
        print("Student: (I have not learned anything yet -- run me with "
              "--host <ollama-ip> so a teacher can train me)")
    else:
        print("Student: (nothing to say about that yet)")
    if cfg.debug:
        print("  class: predicted={0} target={1} p={2:.3f}".format(
            report["class"]["predicted"], report["class"]["target"],
            report["class"]["probability"]))
        print("  experts: {0} | reward {1:+.3f}".format(
            report["experts"], report["reward"]))
        print("  breakdown: " + ", ".join(
            "{0}={1:+.2f}".format(k, v)
            for k, v in sorted(report["reward_breakdown"].items())))
        print("  retrained: {0} | lifecycle: {1}".format(
            [r["expert"] for r in report["retrained"]], report["lifecycle"]))
    return True


def main():
    return run()


if __name__ == "__main__":
    sys.exit(run())
