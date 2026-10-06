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


1. Reference implementation

Load a Hugging Face VLM.
Separate vision encoder, projector, tokenizer, and language model.
Implement prefill and token-by-token decoding manually.
Verify logits against Hugging Face.

2. Core inference runtime

Implement a paged KV cache.
Add continuous batching.
Build a scheduler for prefill and decode requests.
Support sampling, stopping conditions, and streaming.

3. VLM-specific pipeline

Decode and resize images asynchronously.
Batch vision-encoder execution.
Insert projected image embeddings at image-token positions.
Cache image embeddings using content hashes.
Track variable image-token counts when scheduling context capacity.

4. Performance work

Use FlashAttention.
Capture decode paths with CUDA Graphs.
Profile CPU preprocessing, H2D copies, prefill, and decode separately.
Add tensor parallelism only after single-GPU execution is solid.
Later consider quantization, prefix caching, and chunked prefill.