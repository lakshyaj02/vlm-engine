from collections import deque
from dataclasses import dataclass, field

import torch

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

        for state in list(self.running):
            # Decode step for running requests
            self.model.decode_request(state=state)
            if state.is_finished():
                self.running.remove(state)
                self.completed.append(state)

        if new_state is not None:
            self.running.append(new_state)

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)