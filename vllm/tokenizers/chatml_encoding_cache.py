# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded reuse of independently encoded ChatML segments.

Inspired by upstream PRs 45514 (segment reuse) and 47583 (renderer integration).
The supported tokenizer is immutable for a renderer's lifetime. Observable
backend changes invalidate this cache; replacement requires a fresh renderer.
"""

from __future__ import annotations

import json
import sys
from array import array
from collections import OrderedDict
from threading import Lock
from typing import Any

_DELIMITER = "<|im_end|>"


def _state(component: Any) -> bytes | None:
    if isinstance(component, dict):
        return json.dumps(component, sort_keys=True).encode()
    return None if component is None else component.__getstate__()


def _signature(backend: Any) -> tuple:
    if type(backend).__module__.partition(".")[0] == "fastokens":
        # The fastokens compatibility shim retains immutable JSON; its added
        # vocabulary getter parses the entire vocabulary on every call.
        return (
            backend._json,
            backend.get_vocab_size(),
            _state(backend.post_processor),
            backend.encode_special_tokens,
        )
    return (
        # Added entries are checked separately. tokenizers 0.22.x can clone
        # the full vocabulary when with_added_tokens=True, costing O(vocab).
        backend.get_vocab_size(with_added_tokens=False),
        tuple(
            (index, token.__getstate__())
            for index, token in sorted(backend.get_added_tokens_decoder().items())
        ),
        _state(backend.normalizer),
        _state(backend.pre_tokenizer),
        _state(backend.post_processor),
    )


class ChatMLEncodingCache:
    """Return complete token IDs, or None to request the original encode path."""

    def __init__(
        self,
        tokenizer: Any,
        delimiter_id: int,
        *,
        max_bytes: int,
        max_entries: int = 2048,
        min_chars: int = 4096,
    ) -> None:
        self.tokenizer = tokenizer
        self._backend = tokenizer.backend_tokenizer
        self._signature = _signature(self._backend)
        self.delimiter_id = delimiter_id
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self.min_chars = min_chars
        self._entries: OrderedDict[tuple[str | None, str], tuple[bytes, int]] = (
            OrderedDict()
        )
        self._lock = Lock()
        self._invalidated = False
        self._bytes = 0
        self._hits = self._misses = self._evictions = self._requests = 0

    @classmethod
    def create(cls, tokenizer: Any, **kwargs: Any) -> ChatMLEncodingCache | None:
        """Enable only the verified BPE/ByteLevel ChatML encoding contract."""
        backend = getattr(tokenizer, "backend_tokenizer", None)
        if backend is None or getattr(tokenizer, "split_special_tokens", False):
            return None
        try:
            config = json.loads(backend.to_str())
            added = config["added_tokens"]
            marker = next(t for t in added if t["content"] == _DELIMITER)
            if not marker["special"] or any(
                marker[k] for k in ("normalized", "single_word", "lstrip", "rstrip")
            ):
                return None
            for token in added:
                content = token["content"]
                if content == _DELIMITER:
                    continue
                # A competing added token must not consume any portion of a
                # boundary from the left, the right, or around the whole marker.
                if _DELIMITER in content or any(
                    content.endswith(_DELIMITER[:n])
                    or content.startswith(_DELIMITER[-n:])
                    for n in range(1, len(_DELIMITER))
                ):
                    return None
            model = config["model"]
            if model["type"] != "BPE" or model.get("dropout"):
                return None
            if config.get("normalizer") not in (None, {"type": "NFC"}):
                return None
            post = config.get("post_processor")
            if post is not None and post.get("type") != "ByteLevel":
                return None
            if tokenizer.num_special_tokens_to_add() != 0:
                return None
            if config.get("decoder", {}).get("type") != "ByteLevel":
                return None
            return cls(tokenizer, marker["id"], **kwargs)
        except (AttributeError, KeyError, TypeError, ValueError, StopIteration):
            return None

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "entries": len(self._entries),
                "accounted_bytes": self._bytes,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "requests": self._requests,
            }

    def _unchanged(self) -> bool:
        if self._invalidated:
            return False
        try:
            unchanged = (
                self.tokenizer.backend_tokenizer is self._backend
                and _signature(self._backend) == self._signature
                and not getattr(self.tokenizer, "split_special_tokens", False)
                and self.tokenizer.num_special_tokens_to_add() == 0
            )
        except (AttributeError, TypeError, ValueError):
            unchanged = False
        if not unchanged:
            with self._lock:
                self._entries.clear()
                self._bytes = 0
                self._invalidated = True
        return unchanged

    def encode(
        self,
        text: str,
        *,
        cache_salt: str | None = None,
        add_special_tokens: bool = True,
        truncation: bool = False,
        max_length: int | None = None,
        **kwargs: Any,
    ) -> list[int] | None:
        """Reuse exact segment encodings without changing the full token stream."""
        if (
            kwargs
            or not isinstance(text, str)
            or len(text) < self.min_chars
            or _DELIMITER not in text
            or type(add_special_tokens) is not bool
            or type(truncation) is not bool
            or (cache_salt is not None and not isinstance(cache_salt, str))
            or (
                max_length is not None
                and (type(max_length) is not int or max_length <= 0)
            )
            or (truncation and max_length is None)
            or self.tokenizer.truncation_side not in ("left", "right")
            or not self._unchanged()
        ):
            return None
        parts = text.split(_DELIMITER)
        found: dict[str, bytes] = {}
        missing: dict[str, None] = {}
        with self._lock:
            for part in parts:
                key = (cache_salt, part)
                entry = self._entries.get(key)
                if entry is None:
                    missing[part] = None
                    self._misses += 1
                else:
                    self._entries.move_to_end(key)
                    found[part] = entry[0]
                    self._hits += 1
        if missing:
            encoded = self.tokenizer(
                list(missing),
                add_special_tokens=False,
                truncation=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )["input_ids"]
            for part, ids in zip(missing, encoded, strict=True):
                packed = array("I", ids).tobytes()
                found[part] = packed
                key = (cache_salt, part)
                # Account retained Python payloads and a conservative LRU-node
                # allowance. This is a cache budget, not a process RSS limit.
                charge = (
                    sys.getsizeof(key)
                    + sys.getsizeof(part)
                    + sys.getsizeof(cache_salt)
                    + sys.getsizeof(packed)
                    + 160
                )
                if charge > self.max_bytes or self.max_entries <= 0:
                    continue
                with self._lock:
                    if self._invalidated:
                        continue
                    old = self._entries.pop(key, None)
                    if old is not None:
                        self._bytes -= old[1]
                    self._entries[key] = (packed, charge)
                    self._bytes += charge
                    while (
                        self._bytes > self.max_bytes
                        or len(self._entries) > self.max_entries
                    ):
                        _, (_, cost) = self._entries.popitem(last=False)
                        self._bytes -= cost
                        self._evictions += 1
        result: list[int] = []
        for index, part in enumerate(parts):
            if index:
                result.append(self.delimiter_id)
            ids_array = array("I")
            ids_array.frombytes(found[part])
            result.extend(ids_array)
        if truncation and max_length is not None and len(result) > max_length:
            if self.tokenizer.truncation_side == "left":
                result = result[-max_length:]
            else:
                result = result[:max_length]
        with self._lock:
            self._requests += 1
        return result
