# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import pytest

from vllm import SamplingParams
from vllm.exceptions import VLLMValidationError


@dataclass
class MockModelConfig:
    is_diffusion: bool = False
    max_logprobs: int = 20
    logits_processors: list | None = None

    def get_vocab_size(self) -> int:
        return 1024


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": 0.7},
        {"temperature": 0.0},
        {"min_p": 0.1},
        {"seed": 42},
        {"min_tokens": 5},
        {"logit_bias": {0: 1.0}},
        {"bad_words": ["foo"]},
        {"allowed_token_ids": [0, 1]},
    ],
)
def test_diffusion_rejects_unsupported_params(kwargs: dict):
    params = SamplingParams(**kwargs)
    with pytest.raises(VLLMValidationError, match="not yet supported with diffusion"):
        params.verify(MockModelConfig(is_diffusion=True), None, None, None)


def test_diffusion_accepts_default_params():
    SamplingParams().verify(MockModelConfig(is_diffusion=True), None, None, None)


def test_diffusion_accepts_top_k_top_p():
    params = SamplingParams(top_p=0.9, top_k=10)
    params.verify(MockModelConfig(is_diffusion=True), None, None, None)


def test_non_diffusion_models_unaffected():
    params = SamplingParams(temperature=0.7, top_k=10, seed=42)
    params.verify(MockModelConfig(), None, None, None)


@pytest.mark.parametrize("positions", [[], [0], [-1], [True], [1.5], [2, 1], [1, 1]])
def test_prompt_logprob_positions_reject_ambiguous_targets(positions):
    with pytest.raises(VLLMValidationError, match="prompt_logprob_positions"):
        SamplingParams(prompt_logprobs=0, prompt_logprob_positions=positions)


def test_prompt_logprob_positions_require_scoring():
    with pytest.raises(VLLMValidationError, match="requires prompt_logprobs"):
        SamplingParams(prompt_logprob_positions=[1, 4])


@pytest.mark.parametrize("skip", [None, False, True])
def test_prompt_logprob_positions_allow_bounded_cache_unless_disabled(skip):
    params = SamplingParams(
        prompt_logprobs=0,
        prompt_logprob_positions=[1, 4],
        skip_reading_prefix_cache=skip,
    )
    assert params.skip_reading_prefix_cache is (skip is True)
    assert SamplingParams(prompt_logprobs=0).skip_reading_prefix_cache is True


def test_prompt_logprob_positions_survive_request_serialization():
    import msgspec

    params = SamplingParams.from_optional(
        prompt_logprobs=0, prompt_logprob_positions=[1, 4, 8]
    )
    restored = msgspec.msgpack.decode(
        msgspec.msgpack.encode(params), type=SamplingParams
    )
    assert restored.prompt_logprob_positions == [1, 4, 8]
    assert restored.skip_reading_prefix_cache is False
