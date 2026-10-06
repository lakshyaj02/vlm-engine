from dataclasses import dataclass
from typing import Any

import torch

@dataclass
class SequenceState:
    request_id: str
    prompt_token_ids: torch.Tensor
    generated_token_ids: torch.Tensor
    attention_mask: torch.Tensor
    past_key_values: Any
    max_new_tokens: int
    eos_token_ids: torch.Tensor
    finished: bool = False

    def __post_init__(self) -> None:
        if self.prompt_token_ids.shape[0] != 1:
            raise ValueError("SequenceState supports exactly one sequence")
        if self.generated_token_ids.shape[0] != 1:
            raise ValueError("SequenceState supports exactly one sequence")
        if self.attention_mask.shape[0] != 1:
            raise ValueError("SequenceState supports exactly one sequence")

    def append_token(self, token_id: torch.Tensor) -> None:
        self.generated_token_ids = torch.cat([self.generated_token_ids, token_id.unsqueeze(1)], dim=1)
        self.attention_mask = torch.cat([self.attention_mask, torch.ones_like(token_id.unsqueeze(1))], dim=1)
        if torch.isin(token_id, self.eos_token_ids).item():
            self.finished = True

        if self.generated_token_ids.size(1) >= self.max_new_tokens:
            self.finished = True

    def is_finished(self) -> bool:
        return self.finished

    def next_input_token(self) -> torch.Tensor:
        return self.generated_token_ids[:, -1] if self.generated_token_ids.size(1) > 0 else self.prompt_token_ids[:, -1]

    def can_generate_more(self) -> bool:
        return self.generated_token_ids.size(1) < self.max_new_tokens and not self.finished
