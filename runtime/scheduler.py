from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import torch
from transformers.modeling_outputs import CausalLMOutputWithPast

from runtime.model import LlavaReferenceModel
from runtime.request import SequenceState

@dataclass
class PendingRequest:
    state: SequenceState
    pixel_values: torch.Tensor


@dataclass
class Scheduler:
    model: LlavaReferenceModel
    waiting: deque[PendingRequest] = field(default_factory=deque)
    running: deque[SequenceState] = field(default_factory=deque)
    completed: deque[SequenceState] = field(default_factory=deque)
    max_batch_size: int = 8

    def add_request(
        self,
        state: SequenceState,
        pixel_values: torch.Tensor,
    ) -> None:
        if state.is_finished():
            self.completed.append(state)
        else:
            self.waiting.append(PendingRequest(state=state, pixel_values=pixel_values))

    def step(self) -> None:
        new_state = None
        if len(self.waiting):
            pending_request = self.waiting.popleft()
            state = pending_request.state
            self.model.prefill_request(state=state, pixel_values=pending_request.pixel_values)
            if state.is_finished():
                self.completed.append(state)
            else:
                new_state = state

        # Group running request by same attention mask size
        groups: dict[int, list[SequenceState]] = {}
        for state in self.running:
            seq_len = state.attention_mask.size(1)
            groups.setdefault(seq_len, []).append(state)

        for states in groups.values():
            batch = DecodeBatch.build_decode_batch(states)
            output = self.model.decode_one(
                            next_token_id=batch.input_ids,
                            attention_mask=batch.attention_mask,
                            past_key_values=batch.past_key_values
                        )
            batch.apply_decode_output(output)


        for state in list(self.running):
            if state.is_finished():
                self.running.remove(state)
                self.completed.append(state)

        if new_state is not None:
            self.running.append(new_state)

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

@dataclass
class DecodeBatch:
    states: list[SequenceState]
    input_ids: torch.Tensor       # [batch, 1]
    attention_mask: torch.Tensor  # [batch, sequence_length]
    past_key_values: Any

    @classmethod
    def build_decode_batch(cls, states: list[SequenceState]) -> DecodeBatch:
        if not states:
            raise ValueError("No states provided")
        # Check every state can generate
        for state in states:
            if not state.can_generate_more():
                raise ValueError(f"State {state} cannot generate")

        # Check length
        seq_lengths = [state.attention_mask.size(1) for state in states]
        if len(set(seq_lengths)) != 1:
            raise ValueError("All states must have the same sequence length")

        input_ids = torch.cat([state.next_input_token().unsqueeze(-1) for state in states], dim=0)
        attention_mask = torch.cat([state.attention_mask for state in states], dim=0)

        cache_type = type(states[0].past_key_values)
        past_key_values = cache_type.from_legacy_cache(tuple(
            tuple(
                torch.cat(
                    [state.past_key_values[layer][cache_index] for state in states],
                    dim=0,
                )
                for cache_index in range(len(states[0].past_key_values[layer]))
            )
            for layer in range(len(states[0].past_key_values))
        ))

        return DecodeBatch(
            states=states,
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )

    def apply_decode_output(
        self,
        output: CausalLMOutputWithPast,
    ) -> None:
        next_tokens = output.logits[:, -1, :]  # [batch, vocab_size]
        next_tokens = next_tokens.argmax(dim=-1)  # [batch]
        cache_type = type(output.past_key_values)
        for index, state in enumerate(self.states):
            state.past_key_values = cache_type.from_legacy_cache(tuple(
                tuple(
                    cache_tensor[index : index + 1]
                    for cache_tensor in output.past_key_values[layer]
                )
                for layer in range(len(output.past_key_values))
            ))
            state.append_token(next_tokens[index : index + 1])
