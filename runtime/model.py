from __future__ import annotations

from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers.modeling_outputs import CausalLMOutputWithPast

from runtime.request import SequenceState

class LlavaReferenceModel:
    """A correctness-first LLaVA runtime with explicit prefill and decode steps."""

    def __init__(self, model: LlavaForConditionalGeneration, processor: Any) -> None:
        self.model = model.eval()
        self.processor = processor
        self.eos_token_id = model.config.eos_token_id

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ) -> "LlavaReferenceModel":
        processor = AutoProcessor.from_pretrained(model_id)
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
        ).to(device)
        return cls(model, processor)

    @property
    def tokenizer(self) -> Any:
        return self.processor.tokenizer

    @property
    def vision_encoder(self) -> torch.nn.Module:
        return self.model.vision_tower

    @property
    def projector(self) -> torch.nn.Module:
        return self.model.multi_modal_projector

    @property
    def language_model(self) -> torch.nn.Module:
        return self.model.language_model

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def prepare_inputs(self, prompt: str, image: Image.Image) -> dict[str, torch.Tensor]:
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        return {name: tensor.to(self.device) for name, tensor in inputs.items()}

    @torch.inference_mode()
    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vision_outputs = self.vision_encoder(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )
        feature_layer = self.model.config.vision_feature_layer
        if not isinstance(feature_layer, int):
            raise NotImplementedError("Only one vision feature layer is supported")

        image_features = vision_outputs.hidden_states[feature_layer]
        strategy = self.model.config.vision_feature_select_strategy
        if strategy == "default":
            image_features = image_features[:, 1:]
        elif strategy != "full":
            raise ValueError(f"Unknown vision feature selection strategy: {strategy}")

        return self.projector(image_features)

    @torch.inference_mode()
    def prefill(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Any:
        token_embeddings = self.language_model.get_input_embeddings()(input_ids)
        image_features = self.encode_image(pixel_values).to(
            device=token_embeddings.device,
            dtype=token_embeddings.dtype,
        )

        image_mask = (input_ids == self.model.config.image_token_index).unsqueeze(-1)
        image_mask = image_mask.expand_as(token_embeddings)
        expected_values = int(image_mask.sum().item())
        if expected_values != image_features.numel():
            raise ValueError(
                "The number of image-token embedding slots does not match the "
                f"projected image features ({expected_values} != {image_features.numel()})"
            )

        inputs_embeds = token_embeddings.masked_scatter(image_mask, image_features)
        decoder_output = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
        return CausalLMOutputWithPast(
            logits=self.model.lm_head(decoder_output.last_hidden_state),
            past_key_values=decoder_output.past_key_values,
            hidden_states=decoder_output.hidden_states,
            attentions=decoder_output.attentions,
        )

    @torch.inference_mode()
    def decode_one(
        self,
        next_token_id: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: Any,
    ) -> Any:
        decoder_output = self.language_model(
            input_ids=next_token_id,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        return CausalLMOutputWithPast(
            logits=self.model.lm_head(decoder_output.last_hidden_state),
            past_key_values=decoder_output.past_key_values,
            hidden_states=decoder_output.hidden_states,
            attentions=decoder_output.attentions,
        )

    @torch.inference_mode()
    def generate(self,
                input_ids: torch.Tensor,
                pixel_values: torch.Tensor,
                attention_mask: torch.Tensor,
                max_new_tokens: int) -> torch.Tensor:

        if max_new_tokens <= 0:
            return input_ids.clone()

        if self.eos_token_id is None:
            raise ValueError("eos_token_id must be set for generation.")

        eos_ids = torch.as_tensor(
            self.eos_token_id,
            device=input_ids.device,
            dtype=input_ids.dtype,
        ).flatten() 
        
        prefill_output = self.prefill(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
        )

        first_token = torch.argmax(prefill_output.logits[:, -1, :], dim=-1)
        generated_tokens = first_token.unsqueeze(1)
        past_key_values = prefill_output.past_key_values
        finished = torch.isin(first_token, eos_ids)

        for _ in range(max_new_tokens - 1):
            token = generated_tokens[:, -1]
            if finished.all():
                break
            else:
                attention_mask = torch.cat([attention_mask, torch.ones_like(token.unsqueeze(1))], dim=1)
                next_output = self.decode_one(
                    next_token_id=token.unsqueeze(-1),
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                )
                next_token = torch.argmax(next_output.logits[:, -1, :], dim=-1)
                decode_token = torch.where(
                    finished,
                    torch.full_like(next_token, self.model.config.pad_token_id),
                    next_token
                )
                finished |= torch.isin(decode_token, eos_ids)
                generated_tokens = torch.cat([generated_tokens, decode_token.unsqueeze(1)], dim=1)
                past_key_values = next_output.past_key_values 

        return torch.cat([input_ids, generated_tokens], dim=1)

    def prefill_request(self, state: SequenceState, pixel_values: torch.Tensor) -> SequenceState:
        if state.max_new_tokens <= 0:
            return state
        
        input_ids = state.prompt_token_ids
        attention_mask = state.attention_mask
        output =  self.prefill(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
        )
        first_token = torch.argmax(output.logits[:, -1, :], dim=-1)
        state.past_key_values = output.past_key_values
        state.append_token(first_token)
        return state

    def decode_request(self, state: SequenceState) -> SequenceState:
        if state.is_finished():
            return state
        output = self.decode_one(
            next_token_id=state.next_input_token().unsqueeze(-1),
            attention_mask=state.attention_mask,
            past_key_values=state.past_key_values,
        )
        state.past_key_values = output.past_key_values
        state.append_token(output.logits[:, -1, :].argmax(dim=-1))
        return state
