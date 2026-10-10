"""Content digests: SHA-256 of what a value *contains*, not of the label it carries (V3; design §2.3, §4.1).

``semantic_digest`` walks dataclasses (fields with ``compare=False`` are derived or bookkeeping and skipped), mappings
(entries sorted by the repr of their keys), sequences, NumPy arrays (dtype, shape and C-order bytes), and scalars
(floats by repr). Two objects have equal digests exactly when the simulator and the LP would read the same numbers
from them, so V3 compares digests of the consumed content, not stored hash labels.
"""

import dataclasses
import hashlib
from collections.abc import Mapping

import numpy as np


def _feed(h, x) -> None:
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        h.update(b"D" + type(x).__name__.encode() + b"(")
        for f in dataclasses.fields(x):
            if f.compare:
                h.update(f.name.encode() + b"=")
                _feed(h, getattr(x, f.name))
        h.update(b")")
    elif isinstance(x, np.ndarray):
        a = np.ascontiguousarray(x)
        h.update(b"A" + a.dtype.str.encode() + repr(a.shape).encode())
        h.update(a.tobytes(order="C"))
    elif isinstance(x, Mapping):
        h.update(b"M{")
        for k in sorted(x, key=repr):
            h.update(repr(k).encode() + b":")
            _feed(h, x[k])
        h.update(b"}")
    elif isinstance(x, (list, tuple)):
        h.update(b"L[")
        for v in x:
            _feed(h, v)
            h.update(b",")
        h.update(b"]")
    elif isinstance(x, float):
        h.update(b"F" + repr(x).encode())
    elif isinstance(x, (bool, int, str, type(None), np.integer, np.floating, np.bool_)):
        h.update(type(x).__name__.encode() + b":" + repr(x).encode())
    else:
        raise TypeError(f"semantic_digest cannot encode {type(x).__name__}")


def semantic_digest(obj) -> str:
    """SHA-256 hex digest of ``obj``'s content (dataclass fields with ``compare=False`` excluded)."""
    h = hashlib.sha256()
    _feed(h, obj)
    return h.hexdigest()


_FIELD_DIGESTS: dict[int, tuple[object, bytes]] = {}  # id of a field's value -> (the value, kept alive; its digest)


def instance_digest(inst) -> str:
    """The same identity as ``semantic_digest(inst)`` gives (equal content, equal digest), faster across a rolling window.

    A rolled instance shares most of its big fields with its parent (``dataclasses.replace`` copies references), so each
    big field's digest is kept by the identity of its value and only fields that are new objects are walked again. The
    string differs from ``semantic_digest``'s; it is only ever compared with another ``content_digest``.
    """
    h = hashlib.sha256(b"Instance(")
    for f in dataclasses.fields(inst):
        if not f.compare:
            continue
        v = getattr(inst, f.name)
        h.update(f.name.encode() + b"=")
        if isinstance(v, (tuple, dict, np.ndarray)) or (dataclasses.is_dataclass(v) and not isinstance(v, type)):
            hit = _FIELD_DIGESTS.get(id(v))
            if hit is None or hit[0] is not v:
                sub = hashlib.sha256()
                _feed(sub, v)
                hit = (v, sub.digest())
                _FIELD_DIGESTS[id(v)] = hit
            h.update(hit[1])
        else:
            _feed(h, v)
    h.update(b")")
    return h.hexdigest()
