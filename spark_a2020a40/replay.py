# -*- coding: utf-8 -*-
"""Bounded replay buffers with a deterministic train/validation split.

Every buffer in the system has a hard capacity.  The specification forbids
unbounded caches, and on a 1 GB phone an unbounded list of feature vectors is
the fastest way to get the process killed by the OOM reaper.
"""

from collections import deque


class Sample(object):
    """One training example: features, target, reward and bookkeeping."""

    __slots__ = ("features", "target", "reward", "step", "tag")

    def __init__(self, features, target, reward=0.0, step=0, tag=""):
        self.features = list(features)
        self.target = 1 if target else 0
        self.reward = float(reward)
        self.step = int(step)
        self.tag = tag

    def as_pair(self):
        return (self.features, self.target)

    def to_dict(self, digits=4):
        return {
            "f": [round(float(v), digits) for v in self.features],
            "t": self.target,
            "r": round(self.reward, digits),
            "s": self.step,
            "g": self.tag,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(data.get("f") or [], data.get("t", 0),
                   data.get("r", 0.0), data.get("s", 0), data.get("g", ""))


class ReplayBuffer(object):
    """Fixed-capacity ring buffer (oldest entries fall off the back)."""

    __slots__ = ("capacity", "_items", "_added")

    def __init__(self, capacity=64):
        self.capacity = max(1, int(capacity))
        self._items = deque(maxlen=self.capacity)
        self._added = 0

    def __len__(self):
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def add(self, sample):
        self._items.append(sample)
        self._added += 1
        return sample

    def add_example(self, features, target, reward=0.0, step=0, tag=""):
        return self.add(Sample(features, target, reward, step, tag))

    def extend(self, samples):
        for sample in samples:
            self.add(sample)
        return self

    def clear(self):
        self._items.clear()

    @property
    def total_added(self):
        return self._added

    def items(self):
        return list(self._items)

    def pairs(self):
        return [s.as_pair() for s in self._items]

    def class_balance(self):
        ones = sum(1 for s in self._items if s.target)
        total = len(self._items)
        return (0.5 if total == 0 else float(ones) / float(total))

    def split(self, validation_fraction=0.25):
        """Deterministic split -- no RNG, so validation is reproducible.

        Every ``stride``-th sample goes to validation, which mixes recent and
        older examples in both halves instead of validating only on the tail.
        """
        items = list(self._items)
        if len(items) < 4 or validation_fraction <= 0.0:
            return items, []
        stride = max(2, int(round(1.0 / validation_fraction)))
        train = []
        validation = []
        for index, sample in enumerate(items):
            if index % stride == stride - 1:
                validation.append(sample)
            else:
                train.append(sample)
        if not validation or not train:
            return items, []
        return train, validation

    def recent(self, count):
        items = list(self._items)
        return items[-max(0, int(count)):]

    def mixed_batch(self, count):
        """Half recent, half oldest -- cheap protection from catastrophic
        forgetting without storing a second buffer."""
        items = list(self._items)
        count = min(len(items), max(1, int(count)))
        half = count // 2
        if half == 0:
            return items[-count:]
        return items[:count - half] + items[-half:]

    def to_dict(self, limit=None, digits=4):
        items = list(self._items)
        if limit is not None:
            items = items[-int(limit):]
        return {"capacity": self.capacity, "added": self._added,
                "items": [s.to_dict(digits) for s in items]}

    @classmethod
    def from_dict(cls, data, capacity=None):
        buffer = cls(capacity or (data or {}).get("capacity", 64))
        for entry in (data or {}).get("items") or []:
            buffer.add(Sample.from_dict(entry))
        buffer._added = int((data or {}).get("added", len(buffer)))
        return buffer
