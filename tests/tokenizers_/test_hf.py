# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pickle
from copy import deepcopy

import pytest
from transformers import AutoTokenizer

from vllm.tokenizers import TokenizerLike
from vllm.tokenizers.hf import (
    ThreadSafeHFTokenizerMixin,
    get_cached_tokenizer,
    maybe_make_thread_pool,
)


@pytest.mark.parametrize("model_id", ["openai-community/gpt2", "zai-org/chatglm3-6b"])
def test_cached_tokenizer(model_id: str):
    reference_tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True
    )
    reference_tokenizer.add_special_tokens({"cls_token": "<CLS>"})
    reference_tokenizer.add_special_tokens({"additional_special_tokens": ["<SEP>"]})

    cached_tokenizer = get_cached_tokenizer(deepcopy(reference_tokenizer))
    _check_consistency(cached_tokenizer, reference_tokenizer)

    pickled_tokenizer = pickle.dumps(cached_tokenizer)
    unpickled_tokenizer = pickle.loads(pickled_tokenizer)
    _check_consistency(unpickled_tokenizer, reference_tokenizer)


def _check_consistency(target: TokenizerLike, expected: TokenizerLike):
    assert isinstance(target, type(expected))

    # Cached attributes
    assert target.all_special_ids == expected.all_special_ids
    assert target.all_special_tokens == expected.all_special_tokens
    assert target.get_vocab() == expected.get_vocab()
    assert len(target) == len(expected)

    # Other attributes
    assert getattr(target, "padding_side", None) == getattr(
        expected, "padding_side", None
    )

    assert target.encode("prompt") == expected.encode("prompt")


@pytest.mark.parametrize("model_id", ["openai-community/gpt2"])
def test_thread_pool_tokenizer_pickle(model_id: str):
    """Regression test for issue #45433: the thread-pool tokenizer wrapper
    reconstructs through maybe_make_thread_pool on unpickling, which used to
    fall off the end and return None."""
    reference_tokenizer = AutoTokenizer.from_pretrained(model_id)

    pooled_tokenizer = maybe_make_thread_pool(deepcopy(reference_tokenizer))
    assert pooled_tokenizer is not None
    assert isinstance(pooled_tokenizer, ThreadSafeHFTokenizerMixin)

    unpickled_tokenizer = pickle.loads(pickle.dumps(pooled_tokenizer))
    assert unpickled_tokenizer is not None
    assert isinstance(unpickled_tokenizer, ThreadSafeHFTokenizerMixin)
    assert unpickled_tokenizer.encode("prompt") == reference_tokenizer.encode("prompt")

    # Idempotence: wrapping an already-pooled tokenizer returns it unchanged.
    assert maybe_make_thread_pool(pooled_tokenizer) is pooled_tokenizer


@pytest.fixture
def chatml_tokenizer():
    """A local ByteLevel BPE with a genuine non-normalized ChatML boundary."""
    from tokenizers import AddedToken, Tokenizer, decoders, models, normalizers
    from tokenizers.pre_tokenizers import ByteLevel
    from transformers import TokenizersBackend

    alphabet = sorted(ByteLevel.alphabet())
    backend = Tokenizer(models.BPE({s: i for i, s in enumerate(alphabet)}, []))
    backend.pre_tokenizer = ByteLevel(add_prefix_space=False)
    backend.normalizer = normalizers.NFC()
    backend.decoder = decoders.ByteLevel()
    backend.add_special_tokens(
        [AddedToken("<|im_end|>", special=True, normalized=False)]
    )
    return TokenizersBackend(tokenizer_object=backend)


def test_chatml_cache_preserves_ids_and_owns_returned_lists(chatml_tokenizer):
    from vllm.tokenizers.chatml_encoding_cache import ChatMLEncodingCache

    tok = chatml_tokenizer
    cache = ChatMLEncodingCache.create(tok, max_bytes=2048, max_entries=2, min_chars=0)
    assert cache is not None
    text = "e\u0301<|im_end|>\u0301 中文👩🏽‍💻\n<|im_end|><|im_end|>"
    expected = tok(text)["input_ids"]
    assert cache.encode(text) == expected
    cache.encode(text).clear()
    assert cache.encode(text) == expected
    for side in ("left", "right"):
        tok.truncation_side = side
        kwargs = dict(truncation=True, max_length=8)
        assert cache.encode(text, **kwargs) == tok(text, **kwargs)["input_ids"]
    assert cache.snapshot()["evictions"] > 0
    assert cache.snapshot()["accounted_bytes"] <= 2048
    assert cache.encode(text, return_offsets_mapping=True) is None


def test_chatml_cache_rejects_ambiguous_tokens_and_invalidates(chatml_tokenizer):
    from tokenizers import AddedToken

    from vllm.tokenizers.chatml_encoding_cache import ChatMLEncodingCache

    tok = chatml_tokenizer
    cache = ChatMLEncodingCache.create(tok, max_bytes=2048, min_chars=0)
    assert cache.encode("a<|im_end|>b") == tok("a<|im_end|>b")["input_ids"]
    tok.add_tokens([AddedToken("competing<|im_", special=True)])
    assert cache.encode("a<|im_end|>b") is None
    assert cache.snapshot()["entries"] == 0
    assert ChatMLEncodingCache.create(tok, max_bytes=2048) is None


def test_chatml_cache_with_native_pool_and_isolated_salts(chatml_tokenizer):
    from concurrent.futures import ThreadPoolExecutor

    from vllm.tokenizers.chatml_encoding_cache import ChatMLEncodingCache

    tok = chatml_tokenizer
    expected = tok("old<|im_end|>new")["input_ids"]
    cache = ChatMLEncodingCache.create(tok, max_bytes=16384, min_chars=0)
    maybe_make_thread_pool(tok, copies=8)
    cache.encode("old<|im_end|>new", cache_salt="first")
    before = cache.snapshot()["hits"]
    cache.encode("old<|im_end|>new", cache_salt="second")
    assert cache.snapshot()["hits"] == before
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(cache.encode, ["old<|im_end|>new"] * 16))
    assert all(ids == expected for ids in results)
