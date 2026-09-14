# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Differential checks with the actual frozen tokenizer, independent of torch."""

import argparse
import copy
import json
import random

from bench_cpu import Cache


def check(tokenizer_path, backend):
    if backend == "fastokens":
        import fastokens

        fastokens.patch_transformers()
    from tokenizers import AddedToken, normalizers
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    cache = Cache.create(tok, max_bytes=16384, max_entries=8, min_chars=0)
    assert cache is not None
    marker = "<|im_end|>"
    samples = [
        "",
        " ",
        "\n\t",
        "e\u0301",
        "\u0301",
        "中文👩🏽‍💻",
        "<|im_start|>assistant\n",
        "<think>\n",
        "<|endoftext|>",
        '<tool_call>{"path":"a.py"}</tool_call>',
        "\r\n",
        "\x00",
    ]
    rng = random.Random(20260914)
    texts = [marker.join(pair) for pair in ((s, t) for s in samples for t in samples)]
    texts += [
        marker.join(rng.choices(samples, k=rng.randint(2, 10))) for _ in range(80)
    ]
    for text in texts:
        expected = tok(text, add_special_tokens=False)["input_ids"]
        assert cache.encode(text, add_special_tokens=False) == expected
        returned = cache.encode(text)
        returned.clear()
        assert cache.encode(text) == expected, "Caller mutated cached IDs"
    for side in ("left", "right"):
        tok.truncation_side = side
        text = marker.join(samples)
        for length in (1, 8, 10000):
            kwargs = dict(truncation=True, max_length=length, add_special_tokens=False)
            assert cache.encode(text, **kwargs) == tok(text, **kwargs)["input_ids"]
    assert cache.snapshot()["evictions"] > 0
    assert cache.snapshot()["accounted_bytes"] <= 16384
    assert cache.snapshot()["entries"] <= 8
    isolated = Cache.create(tok, max_bytes=16384, min_chars=0)
    text = "Alpha" + marker + "Beta"
    isolated.encode(text, cache_salt="a")
    before = isolated.snapshot()
    isolated.encode(text, cache_salt="b")
    after = isolated.snapshot()
    assert after["hits"] == before["hits"], "Salt isolation failed"
    isolated.encode(text, cache_salt="b")
    assert isolated.snapshot()["hits"] > after["hits"]
    tiny = Cache.create(tok, max_bytes=1, min_chars=0)
    assert tiny.encode(text) == tok(text)["input_ids"]
    assert tiny.snapshot()["entries"] == 0
    assert cache.encode("no boundary") is None
    assert cache.encode(text, return_offsets_mapping=True) is None
    assert cache.encode(text, truncation=True, max_length=0) is None
    tok.split_special_tokens = True
    assert cache.encode(text) is None
    tok.split_special_tokens = False
    # HF supports mutation; fastokens 0.3.1 deliberately makes add_tokens a no-op.
    mutation_checks = 0
    if backend == "hf":
        for mutation in ("add", "normalizer", "marker_flags", "replace"):
            changed = copy.deepcopy(tok)
            guarded = Cache.create(changed, max_bytes=16384, min_chars=0)
            guarded.encode(text)
            if mutation == "add":
                changed.add_tokens(["unique_new_cache_test_token"])
            elif mutation == "normalizer":
                changed.backend_tokenizer.normalizer = normalizers.Lowercase()
            elif mutation == "marker_flags":
                changed.add_tokens([AddedToken(marker, special=True, lstrip=True)])
            else:
                changed._tokenizer = copy.deepcopy(changed._tokenizer)
            assert guarded.encode(text) is None, mutation
            assert guarded.snapshot()["entries"] == 0, mutation
            mutation_checks += 1
        ambiguous = copy.deepcopy(tok)
        ambiguous.add_tokens([AddedToken("prefix<|im_", special=True)])
        assert Cache.create(ambiguous, max_bytes=16384) is None
        mutation_checks += 1
    print(
        json.dumps(
            dict(
                backend=backend,
                boundary_cases=len(texts),
                truncation_cases=6,
                mutation_and_overlap_cases=mutation_checks,
                checks=(
                    "exact IDs, returned-list ownership, "
                    "budget, eviction, salt, fallback"
                ),
                status="PASS",
            )
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--backend", choices=["hf", "fastokens"], default="hf")
    args = parser.parse_args()
    check(args.tokenizer, args.backend)
