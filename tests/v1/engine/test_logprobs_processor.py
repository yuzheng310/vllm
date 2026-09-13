# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for LogprobsProcessor.

These tests exercise the truncation invariant that the MRV2 sampler relies
on: when the sampler returns a row wider than a request's own
`num_logprobs + 1` (because another request in the batch needed a wider
row), the trailing positions are populated with sentinel values
(`token_id=0`, `logprob=-inf`). LogprobsProcessor must read only the first
`num_logprobs + 1` entries so those sentinels never reach the user.
"""

import numpy as np
import pytest
import torch

from vllm.logprobs import create_prompt_logprobs, create_sample_logprobs
from vllm.v1.engine.logprobs import LogprobsProcessor
from vllm.v1.outputs import LogprobsLists, LogprobsTensors


def _make_processor(num_logprobs: int) -> LogprobsProcessor:
    return LogprobsProcessor(
        tokenizer=None,
        logprobs=create_sample_logprobs(flat_logprobs=False),
        prompt_logprobs=None,
        cumulative_logprob=0.0,
        num_logprobs=num_logprobs,
        num_prompt_logprobs=None,
    )


def test_drops_trailing_sentinel_columns():
    """A request that asked for 3 custom token logprobs but ended up in a
    batch padded to width 5 must not surface the trailing -inf entries."""
    processor = _make_processor(num_logprobs=3)

    sampled = 42
    # Layout: [sampled, custom_1, custom_2, custom_3, SENTINEL, SENTINEL]
    # Use float32-exact values so cumulative_logprob compares cleanly.
    token_ids = np.array([[sampled, 100, 200, 300, 0, 0]], dtype=np.int32)
    logprobs = np.array([[-0.5, -1.0, -2.0, -3.0, -np.inf, -np.inf]], dtype=np.float32)
    ranks = np.array([1], dtype=np.int32)

    processor._update_sample_logprobs(LogprobsLists(token_ids, logprobs, ranks))

    assert len(processor.logprobs) == 1
    pos = processor.logprobs[0]
    # Exactly sampled + 3 requested tokens; trailing sentinels dropped.
    assert set(pos.keys()) == {sampled, 100, 200, 300}
    assert 0 not in pos
    assert all(np.isfinite(lp.logprob) for lp in pos.values())
    # cumulative_logprob comes from the sampled token's logprob only.
    assert processor.cumulative_logprob == -0.5


def test_accepts_exactly_sized_row():
    """When the row is exactly num_logprobs+1, no truncation needed."""
    processor = _make_processor(num_logprobs=2)

    token_ids = np.array([[7, 11, 13]], dtype=np.int32)
    logprobs = np.array([[-0.5, -1.5, -2.5]], dtype=np.float32)
    ranks = np.array([1], dtype=np.int32)

    processor._update_sample_logprobs(LogprobsLists(token_ids, logprobs, ranks))

    pos = processor.logprobs[0]
    assert set(pos.keys()) == {7, 11, 13}


@pytest.mark.parametrize("flat", [False, True])
def test_sparse_prompt_scores_keep_original_positions_and_delta_delivery(flat):
    """Empty middle/tail positions must not shift the selected action scores."""
    processor = LogprobsProcessor(
        tokenizer=None,
        logprobs=None,
        prompt_logprobs=create_prompt_logprobs(flat),
        cumulative_logprob=None,
        num_logprobs=None,
        num_prompt_logprobs=0,
        prompt_logprob_positions=[2, 5],
        prompt_token_ids=[10, 11, 12, 13, 14, 15, 16, 17],
    )
    processor._update_prompt_logprobs(
        LogprobsTensors(
            torch.tensor([[12], [15]]),
            torch.tensor([[-0.5], [-1.5]]),
            torch.tensor([1, 2]),
        )
    )
    output = processor.pop_prompt_logprobs()
    assert len(output) == 8
    assert output[2][12].logprob == -0.5
    assert output[5][15].logprob == -1.5
    assert all(not output[i] for i in [0, 1, 3, 4, 6, 7])
    assert not processor.pop_prompt_logprobs()


def test_sparse_prompt_decoding_uses_adjacent_input_tokens(monkeypatch):
    """The previous selected action may be far from the actual UTF-8 context."""
    import vllm.v1.engine.logprobs as module

    contexts = []
    monkeypatch.setattr(
        module, "convert_ids_list_to_tokens", lambda tokenizer, ids: ["x"] * len(ids)
    )

    def verify(self, decoded_tokens_list, tokens, context_token_ids):
        contexts.append(context_token_ids)
        return decoded_tokens_list

    monkeypatch.setattr(LogprobsProcessor, "_verify_tokens", verify)
    processor = LogprobsProcessor(
        tokenizer=object(),
        logprobs=None,
        prompt_logprobs=create_prompt_logprobs(False),
        cumulative_logprob=None,
        num_logprobs=None,
        num_prompt_logprobs=0,
        prompt_logprob_positions=[1, 6],
        prompt_token_ids=list(range(10, 18)),
    )
    processor._update_prompt_logprobs(
        LogprobsTensors(
            torch.tensor([[11], [16]]),
            torch.tensor([[-0.5], [-1.5]]),
            torch.tensor([1, 2]),
        )
    )
    assert contexts == [[10], [12, 13, 14, 15]]
