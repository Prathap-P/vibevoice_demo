# VibeVoice Performance Optimization Roadmap

> **Context**: Mac M4 (Apple Silicon MPS), VibeVoice-Realtime-0.5B, target workload ~50,000 characters.
> Baseline measured: RTF 1.084× at 30 DDPM steps (15.73s audio, 0.40s TTFA).
> Problem: 50k chars ≈ 67 minutes of audio → ~72 minutes generation time at baseline RTF.
> Goal: Bring total generation time under 15 minutes.

---

## The Bottleneck Breakdown

| Component | Share of Compute | Baseline Issue |
|-----------|-----------------|----------------|
| Qwen 2.5-0.5B LLM | ~46% | PyTorch MPS only manages 7–9 tok/s |
| Diffusion Head (DDPM) | ~45% | 30 steps × per-token = bulk of wall time |
| Acoustic Tokenizer VAE | ~9% | Not a bottleneck |

---

## Tier 1 — Immediate Code Changes (No Architecture Change)

**Estimated combined result: RTF ~0.40–0.50× → 67-min audio in ~27–33 min**

### 1A. Reduce DDPM Steps: 30 → 10

The VibeVoice technical report uses **10 steps as the default**. We are running at 3× more than recommended.

```python
# vibevoice_eval.py
DDPM_STEPS = 10   # was 30
```

| Steps | RTF Estimate | 67-min audio | Quality |
|-------|-------------|--------------|---------|
| 30 (current) | 1.084× | ~72 min | Overkill |
| 10 (recommended) | ~0.76× | ~51 min | ✅ Good |
| 5 | ~0.50× | ~33 min | ⚠️ Risk of collapse on long text |

**Source**: VibeVoice Technical Report (arXiv 2508.19205) — "iterative denoising step is 10".

---

### 1B. Switch to bfloat16

M-series chips have native bfloat16 hardware support. This halves memory bandwidth usage and gives ~1.5–2× speedup on memory-bound operations (both LLM and diffusion).

```python
# vibevoice_eval.py — in load_model()
model = VibeVoiceForConditionalGenerationInference.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,     # ADD THIS
    attn_implementation='sdpa',      # ADD THIS (see 1C)
    device_map='auto',
)
```

---

### 1C. Use SDPA Attention

`sdpa` (Scaled Dot-Product Attention) is PyTorch's MPS-native optimized attention. It is faster and more stable on Apple Silicon than the default `eager` mode.

- `flash_attention_2` → **not available on MPS** (CUDA only)
- `sdpa` → ✅ confirmed working on Apple Silicon (from HuggingFace VibeVoice-1.5B discussion)
- `eager` → safe but slowest

```python
attn_implementation='sdpa'
```

---

## Tier 2 — Parallel Segment Processing (Medium Effort, ~1–2 hours)

**Estimated result: 2–4× wall time reduction on top of Tier 1 gains**

VibeVoice's 8k context window supports ~10 minutes of audio per run. 67 minutes of audio requires multiple runs anyway — run them in parallel instead of sequentially.

### Strategy

```
50,000 chars  (split on sentence boundaries)
 ├── Segment 1 (~7,000 chars) → subprocess A → output_seg_1.wav
 ├── Segment 2 (~7,000 chars) → subprocess B → output_seg_2.wav
 ├── Segment 3 (~7,000 chars) → subprocess C → output_seg_3.wav
 ├── ...
 └── Segment 7 (~7,000 chars) → subprocess G → output_seg_7.wav
         ↓
 ffmpeg -i "concat:seg_1.wav|seg_2.wav|..." final_output.wav
```

### Hardware Limits

| Mac Model | RAM | Safe Parallel Processes |
|-----------|-----|------------------------|
| M4 base | 16 GB | 2 |
| M4 Pro | 24–48 GB | 3–4 |
| M4 Max | 48–128 GB | 6–8 |

### Key Rules
- Always split on **sentence boundaries** — never mid-sentence
- Use the **same voice preset `.pt` file** across all processes for voice consistency
- Each subprocess loads its own model copy independently (no shared state)

### Skeleton Code

```python
import multiprocessing, subprocess, textwrap

def synthesise_segment(args):
    seg_idx, text_chunk, voice = args
    out = f"seg_{seg_idx:03d}.wav"
    subprocess.run([
        "python", "vibevoice_eval.py",
        "--text", text_chunk,
        "--voice", voice,
        "--out", out,
    ], check=True)
    return out

def split_sentences(text, max_chars=7000):
    # split on '. ' boundaries, keep under max_chars per chunk
    sentences = text.replace('. ', '.\n').split('\n')
    chunks, buf = [], ''
    for s in sentences:
        if len(buf) + len(s) > max_chars and buf:
            chunks.append(buf.strip())
            buf = s
        else:
            buf += ' ' + s
    if buf:
        chunks.append(buf.strip())
    return chunks

if __name__ == '__main__':
    text = open('input_text.txt').read()
    chunks = split_sentences(text)
    args = [(i, chunk, 'en-Carter_man') for i, chunk in enumerate(chunks)]
    with multiprocessing.Pool(processes=2) as pool:   # adjust for your RAM
        wav_files = pool.map(synthesise_segment, args)
    # stitch with ffmpeg
    concat_list = '|'.join(wav_files)
    subprocess.run(['ffmpeg', '-i', f'concat:{concat_list}', '-c', 'copy', 'final.wav'])
```

---

## Tier 3 — Quantize the Qwen LLM Backbone (Medium Effort)

**Estimated result: ~1.5–2× speedup on the LLM component (46% of compute)**

The LLM is the biggest single compute cost. INT4/INT8 quantization is already proven on VibeVoice — `FabioSarracino/VibeVoice-Large-Q8` uses selective quantization where 52% of params are INT8 but audio-critical components (diffusion head, VAE, connectors) stay at FP32.

### Option A: INT8 (Safer, Start Here)

```python
from transformers import BitsAndBytesConfig

quant_config = BitsAndBytesConfig(load_in_8bit=True)

model = VibeVoiceForConditionalGenerationInference.from_pretrained(
    model_path,
    quantization_config=quant_config,
    torch_dtype=torch.bfloat16,
    attn_implementation='sdpa',
)
```

### Option B: INT4 (More Aggressive)

```python
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
)
```

### Caveats
- `bitsandbytes` MPS support is partial — test INT8 first, INT4 second
- If audio quality degrades, use selective quantization: only quantize LLM layers, keep diffusion head and VAE at FP32
- Reference model for selective approach: `huggingface.co/FabioSarracino/VibeVoice-Large-Q8`

---

## Tier 4 — Architecture Pivot (Longer Term)

### Option A: Port Qwen Backbone to MLX

Research (arXiv 2511.05502) benchmarks on Apple Silicon show:

| Framework | Throughput (Qwen 2.5) | vs PyTorch MPS |
|-----------|----------------------|----------------|
| MLX | ~230 tok/s | **~28× faster** |
| MLC-LLM | ~190 tok/s | ~23× faster |
| llama.cpp | ~150 tok/s | ~18× faster |
| PyTorch MPS | 7–9 tok/s | baseline |

Moving the Qwen backbone to MLX while keeping the diffusion head in PyTorch MPS would require a **tensor bridge** (MLX → numpy → PyTorch) at the LLM output boundary. The bridge adds a small copy overhead but the LLM speedup far outweighs it.

**Complexity**: High. Requires modifying the VibeVoice model forward pass to swap the LLM call.

---

### Option B: Switch to F5-TTS (Flow Matching)

If VibeVoice voice quality is not a hard requirement, F5-TTS uses **flow matching** instead of DDPM diffusion and is dramatically faster:

| Model | RTF | 67-min audio generation | Steps |
|-------|-----|------------------------|-------|
| VibeVoice baseline | 1.084× | ~72 min | 30 DDPM |
| VibeVoice Tier 1+2 optimized | ~0.20× | ~13 min | 10 DDPM + 3× parallel |
| **F5-TTS (16 NFE)** | **0.15×** | **~10 min** | 16 flow steps |
| **F5-TTS Fast (7 NFE)** | **0.030×** | **~2 min** | 7 flow steps |

F5-TTS repo: `github.com/SWivid/F5-TTS`

**Tradeoffs vs VibeVoice**:
- Different voice cloning approach (reference audio, not `.pt` presets)
- Different voice character and expressiveness
- No multi-speaker in a single pass
- But: non-autoregressive, all tokens in parallel, no "per-token diffusion" cost

---

## Combined Target: What Is Realistic on Mac M4

| Optimization Stack | RTF | 67-min audio | Effort |
|-------------------|-----|-------------|--------|
| Baseline (current) | 1.084× | 72 min | — |
| + Steps 10 | ~0.76× | 51 min | 1 line |
| + bfloat16 + SDPA | ~0.45× | 30 min | 2 lines |
| + 3× parallel | ~0.15× | 10 min | ~2 hrs |
| + INT8 quantization | ~0.10× | 7 min | ~1 hr |
| **Full Tier 1–3** | **~0.10×** | **~7 min** | **~3 hrs total** |

---

## Implementation Priority

```
[x] Tier 1A — DDPM_STEPS = 10 in vibevoice_eval.py              (done 2026-05-15)
[x] Tier 1B — torch_dtype=torch.bfloat16 on MPS                 (done 2026-05-15)
[x] Tier 1C — attn_implementation='sdpa' on MPS                 (done 2026-05-14)
[x] Tier 2  — CLI args (--voice, --text, --out, --steps)        (done 2026-05-15)
[ ] Tier 2  — Parallel segment processor + ffmpeg stitch        (2 hrs)
[ ] Tier 3  — INT8 quantization via bitsandbytes                (1 hr)
[ ] Tier 4A — MLX Qwen backbone (research spike needed)         (days)
[ ] Tier 4B — Evaluate F5-TTS as an alternative                 (1 day)
```

---

## References

- [VibeVoice Technical Report — arXiv 2508.19205](https://arxiv.org/html/2508.19205v1)
- [Apple Silicon VibeVoice Script — HuggingFace Discussion](https://huggingface.co/microsoft/VibeVoice-1.5B/discussions/17)
- [Production LLM Inference on Apple Silicon — arXiv 2511.05502](https://arxiv.org/abs/2511.05502)
- [F5-TTS Flow Matching — arXiv 2410.06885](https://arxiv.org/abs/2410.06885)
- [VibeVoice-Large-Q8 Selective Quantization](https://huggingface.co/FabioSarracino/VibeVoice-Large-Q8)
- [VibeVoice Realtime 0.5B Docs](https://github.com/microsoft/VibeVoice/blob/main/docs/vibevoice-realtime-0.5b.md)
