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
                 "last_error", "calls", "_opener", "_models", "_resolved")

    def __init__(self, cfg):
        self.cfg = cfg
        self._available = None
        self._checked_at = 0.0
        self._failures = 0
        self.last_error = None
        self.calls = 0
        self._models = None
        self._resolved = None
        self._opener = urllib.request.build_opener()

    # -- error reporting ------------------------------------------------
    @staticmethod
    def _describe(exc):
        """Turn an exception into something a human can act on.

        Ollama returns the real reason in the body of its 4xx responses
        ("model X not found, try pulling it first").  Reporting only
        "HTTP Error 404" throws that away and leaves the user guessing.
        """
        if isinstance(exc, urllib.error.HTTPError):
            detail = ""
            try:
                raw = exc.read().decode("utf-8", errors="replace")
                if raw:
                    try:
                        detail = json.loads(raw).get("error") or raw
                    except ValueError:
                        detail = raw
            except Exception:
                detail = ""
            detail = (detail or "").strip()
            if detail:
                return "HTTP {0}: {1}".format(exc.code, detail[:300])
            return "HTTP {0}: {1}".format(exc.code, exc.reason)
        if isinstance(exc, urllib.error.URLError):
            return "cannot reach the server: {0}".format(exc.reason)
        return str(exc)

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
            self.last_error = self._describe(exc)
        return self._available

    # -- model discovery -------------------------------------------------
    def list_models(self, refresh=False):
        """Names of the models installed on the Ollama server, or ``None``."""
        if self._models is not None and not refresh:
            return self._models
        try:
            request = urllib.request.Request(
                self.cfg.ollama_url("/api/tags"), headers={"User-Agent": USER_AGENT})
            response = self._opener.open(request, timeout=self.cfg.teacher_probe_timeout)
            raw = response.read().decode("utf-8", errors="replace")
            response.close()
            data = json.loads(raw)
        except Exception as exc:
            self.last_error = self._describe(exc)
            return None
        names = []
        for entry in data.get("models") or []:
            name = entry.get("name") or entry.get("model")
            if name:
                names.append(name)
        self._models = names
        return names

    def resolve_model(self, model=None):
        """Return the *exact* installed name for ``model``, or ``None``.

        Ollama will not accept ``aya-expanse`` when what is installed is
        ``aya-expanse:8b`` -- it answers 404.  Matching on the bare prefix is
        therefore only good enough for a diagnostic, not for the request
        itself, so the resolved name is what actually gets sent.
        """
        wanted = model or self.cfg.ollama_model
        names = self.list_models()
        if not names:
            return None
        if wanted in names:
            return wanted
        if wanted + ":latest" in names:
            return wanted + ":latest"
        stem = wanted.split(":")[0]
        matches = [n for n in names if n.split(":")[0] == stem]
        if len(matches) == 1:
            return matches[0]
        if matches:
            # Ambiguous (several tags): prefer :latest, else the first sorted.
            for candidate in sorted(matches):
                if candidate.endswith(":latest"):
                    return candidate
            return sorted(matches)[0]
        return None

    def check_model(self, model=None):
        """Return ``(ok, message)`` for the configured teacher model.

        A wrong model name is the single most common reason the teacher
        "does not connect" while the server is plainly reachable.
        """
        wanted = model or self.cfg.ollama_model
        names = self.list_models()
        if names is None:
            return True, "could not list models ({0})".format(self.last_error)
        if not names:
            return False, "the Ollama server has no models installed; run: ollama pull " + wanted
        resolved = self.resolve_model(wanted)
        if resolved == wanted:
            return True, "model '{0}' is available".format(wanted)
        if resolved:
            return True, ("'{0}' resolves to the installed '{1}' "
                          "(that exact name will be sent)".format(wanted, resolved))
        return False, ("model '{0}' is not installed on the server. Available: {1}. "
                       "Run: ollama pull {0}".format(wanted, ", ".join(sorted(names))))

    # -- generation -----------------------------------------------------
    def generate(self, prompt, model=None, timeout=None, raise_on_error=False,
                 system=None):
        """Return the teacher's text, or ``None`` when offline.

        ``system`` is passed straight through to Ollama and is how the caller
        pins the reply to one language.
        """
        if not self.cfg.teacher_enabled:
            self.last_error = "teacher disabled by configuration"
            if raise_on_error:
                raise TeacherUnavailable(self.last_error)
            return None
        wanted = model or self.cfg.ollama_model
        if self._resolved is None:
            # Resolved once and cached; falls back to the literal name when
            # the server cannot be listed, so this never blocks offline use.
            self._resolved = self.resolve_model(wanted) or wanted
        payload = {
            "model": self._resolved if not model else (
                self.resolve_model(model) or model),
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system
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
            self.last_error = self._describe(exc)
            # A 4xx is the server talking to us, so it is still reachable --
            # do not mark it offline or we will stop retrying for 30 seconds
            # over what is usually just a bad model name.
            if not isinstance(exc, urllib.error.HTTPError):
                self._available = False
                self._checked_at = time.time()
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
