# VLM Engine

Correctness-first experiments for a vision-language model inference engine.

## Reference implementation

The first milestone targets Hugging Face LLaVA and makes its inference stages
explicit:

- `vision_tower` encodes image patches.
- `multi_modal_projector` maps vision features into the language-model space.
- The processor owns image preprocessing and tokenization.
- `language_model` performs prefill and cached, token-by-token decoding.

Set up the development environment and run the parity test:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
pytest -q
```

The test uses a tiny random LLaVA configuration, so it checks the mechanics and
logits without downloading model weights. Once that passes, load a real model
with `LlavaReferenceModel.from_pretrained(...)`; use a CUDA machine and a small
LLaVA checkpoint for that step.