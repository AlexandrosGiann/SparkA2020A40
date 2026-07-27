# -*- coding: utf-8 -*-
"""Ollama teacher client with first-class offline behaviour.

The legacy code called ``urlopen(..., timeout=120)`` on every single turn, so a
phone that was off the LAN froze for two minutes before printing a traceback.
Here availability is probed cheaply, cached for ``teacher_recheck_seconds``, and
``generate`` degrades to ``None`` instead of raising.  The student never depends
on this module being reachable.
"""

import json
import time
import urllib.error
import urllib.request

USER_AGENT = "SparkA2020A40/2.0 (+student-runtime)"


class TeacherUnavailable(RuntimeError):
    pass


class TeacherClient(object):
    """Thin, timeout-bounded wrapper around ``/api/generate``."""

    __slots__ = ("cfg", "_available", "_checked_at", "_failures",
                 "last_error", "calls", "_opener")

    def __init__(self, cfg):
        self.cfg = cfg
        self._available = None
        self._checked_at = 0.0
        self._failures = 0
        self.last_error = None
        self.calls = 0
        self._opener = urllib.request.build_opener()

    # -- availability ---------------------------------------------------
    def is_available(self, force=False):
        """Cheap cached probe of the Ollama root endpoint."""
        if not self.cfg.teacher_enabled:
            self._available = False
            return False
        now = time.time()
        if (not force and self._available is not None
                and (now - self._checked_at) < self.cfg.teacher_recheck_seconds):
            return self._available
        self._checked_at = now
        try:
            request = urllib.request.Request(
                self.cfg.ollama_url("/"), headers={"User-Agent": USER_AGENT})
            response = self._opener.open(request, timeout=self.cfg.teacher_probe_timeout)
            body = response.read(64)
            response.close()
            self._available = b"Ollama" in body or bool(body)
            self.last_error = None
        except Exception as exc:  # network, DNS, refused, timeout ...
            self._available = False
            self.last_error = str(exc)
        return self._available

    # -- generation -----------------------------------------------------
    def generate(self, prompt, model=None, timeout=None, raise_on_error=False):
        """Return the teacher's text, or ``None`` when offline."""
        if not self.cfg.teacher_enabled:
            self.last_error = "teacher disabled by configuration"
            if raise_on_error:
                raise TeacherUnavailable(self.last_error)
            return None
        payload = {
            "model": model or self.cfg.ollama_model,
            "prompt": prompt,
            "stream": False,
        }
        request = urllib.request.Request(
            self.cfg.ollama_url("/api/generate"),
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
            method="POST",
        )
        try:
            response = self._opener.open(
                request, timeout=timeout or self.cfg.teacher_timeout)
            raw = response.read().decode("utf-8", errors="replace")
            response.close()
        except Exception as exc:
            self._failures += 1
            self._available = False
            self._checked_at = time.time()
            self.last_error = str(exc)
            if raise_on_error:
                raise TeacherUnavailable(self.last_error)
            return None

        self.calls += 1
        self._available = True
        self._failures = 0
        self.last_error = None
        try:
            data = json.loads(raw)
        except ValueError:
            self.last_error = "teacher returned non-JSON payload"
            return None
        text = data.get("response")
        if not isinstance(text, str):
            self.last_error = "teacher response missing 'response' field"
            return None
        return text

    # -- diagnostics -----------------------------------------------------
    def status(self):
        return {
            "enabled": self.cfg.teacher_enabled,
            "url": self.cfg.ollama_url("/api/generate"),
            "model": self.cfg.ollama_model,
            "available": self._available,
            "calls": self.calls,
            "failures": self._failures,
            "last_error": self.last_error,
        }


class OfflineTeacher(TeacherClient):
    """Never touches the network -- used by tests and by ``--offline``."""

    def is_available(self, force=False):
        self._available = False
        return False

    def generate(self, prompt, model=None, timeout=None, raise_on_error=False):
        self.last_error = "offline teacher"
        if raise_on_error:
            raise TeacherUnavailable(self.last_error)
        return None
