import torch
from transformers import CLIPVisionConfig, LlamaConfig, LlavaConfig
from transformers.models.llava.modeling_llava import LlavaForConditionalGeneration
from runtime.request import SequenceState

from runtime.model import LlavaReferenceModel

def make_tiny_llava() -> LlavaForConditionalGeneration:
    vision_config = CLIPVisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        image_size=8,
        patch_size=4,
    )
    text_config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        eos_token_id=2,
        pad_token_id=0,
    )
    config = LlavaConfig(
        vision_config=vision_config,
        text_config=text_config,
        image_token_index=31,
        vision_feature_layer=-2,
        vision_feature_select_strategy="default",
        eos_token_id=2,
        pad_token_id=0,
    )
    return LlavaForConditionalGeneration(config).eval()


def make_sequence_state(
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int = 3,
) -> SequenceState:
    return SequenceState(
        request_id="test-request",
        prompt_token_ids=input_ids,
        generated_token_ids=torch.empty(
            (input_ids.size(0), 0),
            dtype=input_ids.dtype,
            device=input_ids.device,
        ),
        attention_mask=torch.ones_like(input_ids),
        past_key_values=None,
        max_new_tokens=max_new_tokens,
        eos_token_ids=torch.tensor([2], device=input_ids.device),
    )


def test_manual_prefill_and_decode_match_hugging_face() -> None:
    torch.manual_seed(0)
    model = make_tiny_llava()
    reference = LlavaReferenceModel(model, processor=None)

    # An 8x8 image with 4x4 patches produces four projected image tokens.
    input_ids = torch.tensor([[1, 31, 31, 31, 31, 4, 5]])
    attention_mask = torch.ones_like(input_ids)
    pixel_values = torch.randn(1, 3, 8, 8)

    with torch.inference_mode():
        oracle_prefill = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
    manual_prefill = reference.prefill(input_ids, pixel_values, attention_mask)

    torch.testing.assert_close(manual_prefill.logits, oracle_prefill.logits)

    next_token = manual_prefill.logits[:, -1].argmax(dim=-1, keepdim=True)
    decode_attention_mask = torch.cat(
        [attention_mask, torch.ones_like(next_token)], dim=1
    )
    manual_decode = reference.decode_one(
        next_token,
        decode_attention_mask,
        manual_prefill.past_key_values,
    )

    with torch.inference_mode():
        oracle_decode = model(
            input_ids=torch.cat([input_ids, next_token], dim=1),
            pixel_values=pixel_values,
            attention_mask=decode_attention_mask,
            return_dict=True,
        )

    torch.testing.assert_close(
        manual_decode.logits[:, -1],
        oracle_decode.logits[:, -1],
        rtol=1e-4,
        atol=1e-5,
    )

def test_model_generate() -> None:
    torch.manual_seed(0)
    model = make_tiny_llava()
    reference = LlavaReferenceModel(model, processor=None)

    input_ids = torch.tensor([[1, 31, 31, 31, 31, 4, 5]])
    attention_mask = torch.ones_like(input_ids)
    pixel_values = torch.randn(1, 3, 8, 8)

    max_new_tokens = 3
    generated_tokens = reference.generate(
        input_ids=input_ids,
        pixel_values=pixel_values,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
    )

    expected = model.generate(
        input_ids=input_ids,
        pixel_values=pixel_values,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    torch.testing.assert_close(generated_tokens, expected)


def test_sequence_state_next_input_token() -> None:
    torch.manual_seed(0)
    request_id = "test"
    prompt_token_ids = torch.tensor([[1, 2, 3]])
    generated_token_ids = torch.tensor([[4, 5]])
    attention_mask = torch.ones_like(torch.cat([prompt_token_ids, generated_token_ids], dim=1))
    past_key_values = None
    max_new_tokens = 5
    eos_token_ids = torch.tensor([6])

    state = SequenceState(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        generated_token_ids=generated_token_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_token_ids,
    )

    assert torch.equal(state.next_input_token(), generated_token_ids[:, -1])

    state.generated_token_ids = torch.empty((1, 0), dtype=torch.long)
    assert torch.equal(state.next_input_token(), prompt_token_ids[:, -1])


def test_sequence_state_stops_on_eos() -> None:
    state = make_sequence_state(torch.tensor([[1, 2, 3]]), max_new_tokens=5)

    state.append_token(torch.tensor([2]))

    assert state.is_finished()
    assert not state.can_generate_more()
    assert state.generated_token_ids.tolist() == [[2]]
    assert state.attention_mask.shape == (1, 4)


def test_sequence_state_stops_at_token_limit() -> None:
    state = make_sequence_state(torch.tensor([[1, 2, 3]]), max_new_tokens=2)

    state.append_token(torch.tensor([4]))
    assert state.can_generate_more()

    state.append_token(torch.tensor([5]))
    assert state.is_finished()
    assert not state.can_generate_more()


def test_request_prefill_and_decode_match_hugging_face_generate() -> None:
    torch.manual_seed(0)
    model = make_tiny_llava()
    reference = LlavaReferenceModel(model, processor=None)
    input_ids = torch.tensor([[1, 31, 31, 31, 31, 4, 5]])
    pixel_values = torch.randn(1, 3, 8, 8)
    state = make_sequence_state(input_ids, max_new_tokens=3)

    expected = model.generate(
        input_ids=input_ids,
        pixel_values=pixel_values,
        attention_mask=state.attention_mask,
        max_new_tokens=state.max_new_tokens,
        do_sample=False,
    )

    reference.prefill_request(state, pixel_values)
    assert state.past_key_values is not None
    assert state.generated_token_ids.shape == (1, 1)
    assert state.attention_mask.shape == (1, input_ids.size(1) + 1)

    while state.can_generate_more():
        reference.decode_request(state)

    torch.testing.assert_close(
        state.generated_token_ids,
        expected[:, input_ids.size(1):],
    )
    assert state.is_finished()
    assert state.attention_mask.shape == expected.shape


def test_request_methods_leave_non_generating_state_unchanged() -> None:
    torch.manual_seed(0)
    reference = LlavaReferenceModel(make_tiny_llava(), processor=None)
    input_ids = torch.tensor([[1, 31, 31, 31, 31, 4, 5]])
    pixel_values = torch.randn(1, 3, 8, 8)
    state = make_sequence_state(input_ids, max_new_tokens=0)

    returned_state = reference.prefill_request(state, pixel_values)

    assert returned_state is state
    assert state.past_key_values is None
    assert state.generated_token_ids.shape == (1, 0)
    assert torch.equal(state.attention_mask, torch.ones_like(input_ids))

    state.finished = True
    returned_state = reference.decode_request(state)
    assert returned_state is state
    assert state.generated_token_ids.shape == (1, 0)