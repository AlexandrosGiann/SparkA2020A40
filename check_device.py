# -*- coding: utf-8 -*-
"""Device readiness check -- run this FIRST on a new phone.

    python3 check_device.py
    python3 check_device.py --host 192.168.1.29

Answers, in about ten seconds, whether SparkA2020A40 can run here and how fast.
Deliberately written to parse on very old Python 3: no f-strings, no
annotations, no dataclasses, and the interpreter check happens before anything
from the package is imported.
"""

import os
import sys
import time

RESULTS = []


def check(label, fn):
    try:
        detail = fn()
        RESULTS.append((True, label, detail))
        print("  ok    {0:<34} {1}".format(label, detail if detail else ""))
        return True
    except Exception as exc:
        RESULTS.append((False, label, str(exc)))
        print("  FAIL  {0:<34} {1}".format(label, exc))
        return False


# -- 1. interpreter -----------------------------------------------------
def interpreter_version():
    major, minor = sys.version_info[0], sys.version_info[1]
    if major < 3:
        raise RuntimeError("Python 2 is not supported")
    if (major, minor) < (3, 4):
        raise RuntimeError(
            "Python {0}.{1} is below the 3.4 floor (os.replace / casefold / "
            "max(default=) are required)".format(major, minor))
    note = ""
    if (major, minor) < (3, 7):
        note = " (lazy exports in __init__ disabled -- harmless)"
    return "Python {0}.{1}.{2}{3}".format(major, minor, sys.version_info[2], note)


def stdlib_modules():
    missing = []
    for name in ("json", "math", "random", "unicodedata", "tempfile",
                 "collections", "urllib.request", "argparse"):
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
    if missing:
        raise RuntimeError("missing: " + ", ".join(missing))
    return "all present"


def required_builtins():
    problems = []
    if not hasattr(os, "replace"):
        problems.append("os.replace")
    if not hasattr("x", "casefold"):
        problems.append("str.casefold")
    try:
        max([], default=0)
    except TypeError:
        problems.append("max(default=)")
    from collections import deque
    try:
        deque(maxlen=2)
    except TypeError:
        problems.append("deque(maxlen=)")
    import urllib.request
    try:
        urllib.request.Request("http://x/", method="POST")
    except TypeError:
        problems.append("Request(method=)")
    if problems:
        raise RuntimeError("unsupported: " + ", ".join(problems))
    return "os.replace, casefold, deque(maxlen), Request(method)"


def unicode_io():
    text = "καλημέρα κόσμε"
    if text.casefold() != text:
        raise RuntimeError("casefold changed plain lowercase Greek")
    import unicodedata
    if unicodedata.category("ά")[0] != "L":
        raise RuntimeError("unicodedata does not classify Greek as letters")
    encoding = (sys.stdout.encoding or "").lower()
    if encoding and "utf" not in encoding:
        return "stdout is " + encoding + " -- Greek may print as '?'"
    return "Greek letters and NFC are fine"


# -- 2. package ---------------------------------------------------------
def import_package():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import spark_a2020a40
    return "v" + spark_a2020a40.__version__


def atomic_write():
    import tempfile
    from spark_a2020a40.config import Config
    from spark_a2020a40.persistence import Persistence
    directory = tempfile.mkdtemp(prefix="spark-check-")
    path = os.path.join(directory, "probe.json")
    persistence = Persistence(Config(), path)
    if not persistence.save({"memory": {"tokens": {}}}):
        raise RuntimeError(persistence.last_error or "save returned False")
    if persistence.load() is None:
        raise RuntimeError("could not read back what we just wrote")
    leftovers = [n for n in os.listdir(directory) if n.endswith(".tmp")]
    for name in os.listdir(directory):
        os.remove(os.path.join(directory, name))
    os.rmdir(directory)
    if leftovers:
        raise RuntimeError("temp files left behind: " + str(leftovers))
    return "atomic save + reload works on this filesystem"


# -- 3. real work -------------------------------------------------------
def timed_turns(turns=12):
    from spark_a2020a40.config import Config
    from spark_a2020a40.student import StudentModel
    from spark_a2020a40.teacher import OfflineTeacher
    from spark_a2020a40.trainer import TrainingCoordinator

    cfg = Config()
    student = StudentModel(cfg)
    coordinator = TrainingCoordinator(cfg, student, OfflineTeacher(cfg))

    lessons = [
        ("γεια σου", "γεια σου κόσμε πώς είσαι σήμερα"),
        ("hello", "hello there how can i help you today"),
        ("τι κάνεις;", "καλά είμαι ευχαριστώ εσύ τι κάνεις"),
    ]
    train = []
    for index in range(turns):
        prompt, answer = lessons[index % len(lessons)]
        start = time.time()
        coordinator.process_turn(prompt, teacher_text=answer)
        train.append((time.time() - start) * 1000.0)

    infer = []
    for index in range(turns):
        start = time.time()
        student.generate(lessons[index % len(lessons)][0])
        infer.append((time.time() - start) * 1000.0)

    train_mean = sum(train) / len(train)
    infer_mean = sum(infer) / len(infer)
    globals()["_TRAIN_MS"] = train_mean
    globals()["_INFER_MS"] = infer_mean
    return "train {0:.0f} ms/turn, inference {1:.0f} ms/turn".format(
        train_mean, infer_mean)


def peak_memory():
    try:
        import resource
        kilobytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            kilobytes = kilobytes / 1024.0
        globals()["_PEAK_MB"] = kilobytes / 1024.0
        return "{0:.1f} MB peak RSS".format(kilobytes / 1024.0)
    except (ImportError, AttributeError):
        return "resource module unavailable -- cannot measure (not fatal)"


def free_storage():
    try:
        stat = os.statvfs(os.path.dirname(os.path.abspath(__file__)))
        megabytes = (stat.f_bavail * stat.f_frsize) / (1024.0 * 1024.0)
        if megabytes < 20:
            raise RuntimeError("only {0:.0f} MB free".format(megabytes))
        return "{0:.0f} MB free".format(megabytes)
    except AttributeError:
        return "os.statvfs unavailable -- skipped"


def teacher_probe(host, port):
    from spark_a2020a40.config import Config
    from spark_a2020a40.teacher import TeacherClient
    cfg = Config()
    cfg.ollama_host = host
    cfg.ollama_port = port
    client = TeacherClient(cfg)
    if not client.is_available(force=True):
        raise RuntimeError(client.last_error or "no response")
    names = client.list_models(refresh=True)
    ok, message = client.check_model()
    if not ok:
        raise RuntimeError(message)
    if names:
        return "reachable; models: " + ", ".join(sorted(names))
    return "reachable at " + cfg.ollama_url("/")


# -- verdict ------------------------------------------------------------
def main(argv):
    host = None
    port = 11434
    if "--host" in argv:
        host = argv[argv.index("--host") + 1]
    if "--port" in argv:
        port = int(argv[argv.index("--port") + 1])

    print("SparkA2020A40 device check")
    print("=" * 58)
    print("platform: {0}".format(sys.platform))
    print("=" * 58)

    print("\n[1/3] interpreter")
    fatal = not check("python version", interpreter_version)
    fatal = not check("standard library", stdlib_modules) or fatal
    fatal = not check("required builtins", required_builtins) or fatal
    check("unicode / Greek", unicode_io)

    if fatal:
        print("\nVERDICT: this interpreter cannot run the student runtime.")
        print("Install a newer Python 3 (Pydroid 3 is the easiest route).")
        return 2

    print("\n[2/3] package")
    fatal = not check("import spark_a2020a40", import_package) or fatal
    if fatal:
        print("\nVERDICT: the package did not import. Check that this script "
              "sits next to the spark_a2020a40/ directory.")
        return 2
    check("atomic persistence", atomic_write)
    check("free storage", free_storage)

    print("\n[3/3] real work")
    ran = check("12 training + 12 inference turns", timed_turns)
    check("peak memory", peak_memory)
    if host:
        check("ollama teacher", lambda: teacher_probe(host, port))
    else:
        print("  --    ollama teacher                 skipped (pass --host IP)")

    print("\n" + "=" * 58)
    failures = [label for good, label, _ in RESULTS if not good]
    train_ms = globals().get("_TRAIN_MS", 0.0)
    infer_ms = globals().get("_INFER_MS", 0.0)
    peak_mb = globals().get("_PEAK_MB", 0.0)

    if failures:
        print("VERDICT: {0} check(s) failed: {1}".format(
            len(failures), ", ".join(failures)))
        return 1

    print("VERDICT: usable.")
    if ran:
        if infer_ms > 3000:
            print("  Inference is {0:.0f} ms/turn -- painfully slow. Lower "
                  "SPARK_MAX_GEN and SPARK_MAX_CANDIDATES.".format(infer_ms))
        elif infer_ms > 800:
            print("  Inference is {0:.0f} ms/turn -- slow but usable.".format(infer_ms))
        else:
            print("  Inference is {0:.0f} ms/turn -- comfortable.".format(infer_ms))
        print("  Training is {0:.0f} ms/turn.".format(train_ms))
    if peak_mb:
        print("  Peak RSS {0:.1f} MB.".format(peak_mb))
        if peak_mb > 150:
            print("  That is high for a 1 GB device; stay on tiny_android.")
    print("\nNext: python3 -m spark_a2020a40 --offline")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
