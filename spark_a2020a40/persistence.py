# -*- coding: utf-8 -*-
"""Versioned, atomic, self-healing persistence.

The legacy ``save_model`` wrote straight over the live file and ``load_model``
swallowed every exception with a bare ``except:`` that returned an empty model
-- so one interrupted write silently erased everything the student had learned.

This module fixes all three problems:

* **atomic**   -- write to ``<file>.tmp``, ``flush`` + ``fsync``, then
                  ``os.replace`` (atomic on POSIX and on Windows/Android);
* **versioned** -- every payload carries ``schema_version``;
* **recoverable** -- the previous good file is kept as ``<file>.bak`` and used
                  automatically when the primary file is corrupt.

``migrate_v1`` converts the original ``bittreelm_memory.json`` layout, so no
existing memory is lost.
"""

import json
import os
import tempfile
import time

SCHEMA_VERSION = 3
LEGACY_FILE = "bittreelm_memory.json"


class PersistenceError(RuntimeError):
    pass


class CorruptStateError(PersistenceError):
    pass


# ----------------------------------------------------------------------
def _atomic_write(path, payload, compact=True):
    directory = os.path.dirname(os.path.abspath(path)) or "."
    if not os.path.isdir(directory):
        os.makedirs(directory)
    separators = (",", ":") if compact else (", ", ": ")
    handle_fd, temp_path = tempfile.mkstemp(
        prefix=os.path.basename(path) + ".", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(handle_fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, separators=separators)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except Exception:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except OSError:
            pass
        raise
    return path


def validate_state(payload):
    """Raise :class:`CorruptStateError` unless the payload looks like ours."""
    if not isinstance(payload, dict):
        raise CorruptStateError("state is not a JSON object")
    version = payload.get("schema_version")
    if not isinstance(version, int):
        raise CorruptStateError("missing schema_version")
    if version > SCHEMA_VERSION:
        raise CorruptStateError(
            "state schema v{0} is newer than supported v{1}".format(
                version, SCHEMA_VERSION))
    state = payload.get("state")
    if not isinstance(state, dict):
        raise CorruptStateError("missing state object")
    memory = state.get("memory")
    if memory is not None and not isinstance(memory.get("tokens"), dict):
        raise CorruptStateError("memory.tokens is not an object")
    return payload


# ----------------------------------------------------------------------
def is_legacy_payload(data):
    """The v1 file has top-level ``tokens``/``tags`` and no schema_version."""
    if not isinstance(data, dict):
        return False
    if "schema_version" in data:
        return False
    return isinstance(data.get("tokens"), dict) and "tags" in data


def migrate_v1(data, cfg=None):
    """Convert the original BitTreeLM memory into a v2 state dict.

    * ``commonality`` becomes the token count,
    * ``relations`` become order-1 positional statistics,
    * the 7-bit binary ids are decoded back to integers and preserved,
    * legacy ``tags`` are retained under ``legacy.tags`` so nothing is lost.
    """
    if not is_legacy_payload(data):
        raise PersistenceError("payload is not a legacy BitTreeLM memory")

    # The legacy file used "<UNK>"; the v2 runtime uses the lowercase
    # canonical specials, so fold them together instead of keeping both.
    renames = {"<UNK>": "<unk>", "<BOS>": "<bos>", "<EOS>": "<eos>"}

    tokens = {}
    max_id = -1
    for text, entry in (data.get("tokens") or {}).items():
        if not isinstance(entry, dict):
            continue
        text = renames.get(text, text)
        raw_id = entry.get("id", 0)
        try:
            token_id = int(str(raw_id), 2) if isinstance(raw_id, str) else int(raw_id)
        except (TypeError, ValueError):
            token_id = max_id + 1
        max_id = max(max_id, token_id)
        relations = {}
        for target, weight in (entry.get("relations") or {}).items():
            target = renames.get(target, target)
            try:
                relations[target] = int(weight)
            except (TypeError, ValueError):
                continue
        tokens[text] = {
            "id": token_id,
            "k": "special" if text.startswith("<") and text.endswith(">") else "word",
            "c": int(entry.get("commonality", 0) or 0),
            "p1": relations,
            "p2": {},
            "p3": {},
            "c0": 0,
            "c1": 0,
            "e": 0.5,
            "u": 0,
        }

    counts = [entry["c"] for entry in tokens.values()] or [1]
    state = {
        "memory": {
            "tokens": tokens,
            "next_id": max(int(data.get("next_id", 0) or 0), max_id + 1),
            "free_ids": [],
            "clock": 0,
            "max_count": max(1, max(counts)),
            "total_observations": sum(counts),
        },
        "legacy": {
            "tags": dict(data.get("tags") or {}),
            "meta": dict(data.get("meta") or {}),
        },
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": time.time(),
        "migrated_from": 1,
        "state": state,
    }


def migrate_legacy_file(source_path, cfg=None):
    with open(source_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return migrate_v1(data, cfg)


# ----------------------------------------------------------------------
class Persistence(object):
    """Save/load the whole student state next to the phone's storage."""

    __slots__ = ("cfg", "path", "backup_path", "last_error", "saves", "loads")

    def __init__(self, cfg, path=None):
        self.cfg = cfg
        self.path = path or cfg.memory_file
        self.backup_path = self.path + cfg.backup_suffix
        self.last_error = None
        self.saves = 0
        self.loads = 0

    # -- saving ---------------------------------------------------------
    def wrap(self, state):
        return {
            "schema_version": SCHEMA_VERSION,
            "saved_at": time.time(),
            "profile": self.cfg.profile,
            "compact": bool(self.cfg.compact_mode),
            "state": state,
        }

    def save(self, state):
        payload = self.wrap(state)
        try:
            validate_state(payload)
            if os.path.exists(self.path):
                try:
                    _copy_file(self.path, self.backup_path)
                except OSError:
                    pass
            _atomic_write(self.path, payload, compact=self.cfg.compact_mode)
            self.saves += 1
            self.last_error = None
            return True
        except Exception as exc:
            self.last_error = str(exc)
            return False

    # -- loading ---------------------------------------------------------
    def _read(self, path):
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def load(self, allow_legacy_migration=True):
        """Return the state dict, or ``None`` when there is nothing usable.

        Order: primary file -> backup -> legacy ``bittreelm_memory.json``.
        """
        self.last_error = None
        for candidate in (self.path, self.backup_path):
            if not os.path.exists(candidate):
                continue
            try:
                payload = self._read(candidate)
            except (ValueError, OSError) as exc:
                self.last_error = "{0}: {1}".format(candidate, exc)
                continue
            if is_legacy_payload(payload):
                try:
                    payload = migrate_v1(payload, self.cfg)
                except PersistenceError as exc:
                    self.last_error = str(exc)
                    continue
            try:
                validate_state(payload)
            except CorruptStateError as exc:
                self.last_error = "{0}: {1}".format(candidate, exc)
                continue
            self.loads += 1
            return payload["state"]

        if allow_legacy_migration:
            legacy = os.path.join(os.path.dirname(os.path.abspath(self.path)) or ".",
                                  LEGACY_FILE)
            if os.path.exists(legacy):
                try:
                    payload = migrate_legacy_file(legacy, self.cfg)
                    self.loads += 1
                    return payload["state"]
                except (PersistenceError, ValueError, OSError) as exc:
                    self.last_error = "legacy migration failed: {0}".format(exc)
        return None

    # -- diagnostics ------------------------------------------------------
    def size_bytes(self):
        try:
            return os.path.getsize(self.path)
        except OSError:
            return 0

    def status(self):
        return {"path": os.path.abspath(self.path),
                "backup": os.path.abspath(self.backup_path),
                "exists": os.path.exists(self.path),
                "size_bytes": self.size_bytes(),
                "saves": self.saves,
                "loads": self.loads,
                "last_error": self.last_error}


def _copy_file(source, destination):
    with open(source, "rb") as src:
        data = src.read()
    directory = os.path.dirname(os.path.abspath(destination)) or "."
    handle_fd, temp_path = tempfile.mkstemp(dir=directory, suffix=".bakm")
    with os.fdopen(handle_fd, "wb") as dst:
        dst.write(data)
        dst.flush()
        os.fsync(dst.fileno())
    os.replace(temp_path, destination)
    return destination
