# VibeVoice GPU Fix — Zero-Shot Migration Guide
> Exact patches required to run VibeVoice on Apple Silicon (MPS) or CUDA.
> Every item here was discovered through trial and failure on a Mac M4.
> Copy these patterns verbatim — each has a concrete reason it must be exactly this way.

---

## Setup

```bash
# Clone VibeVoice repo alongside your project
git clone https://github.com/microsoft/VibeVoice.git

# Install (streamingtts extra is required — it brings AudioStreamer)
pip install -e './VibeVoice[streamingtts]'

# Download model weights (0.5B only — see MODEL SELECTION below)
huggingface-cli download microsoft/VibeVoice-Realtime-0.5B \
  --local-dir ./models/VibeVoice-Realtime-0.5B

# Voice presets live inside the cloned repo
# ./VibeVoice/demo/voices/streaming_model/<voice_name>.pt
```

---

## FIX 1 — Environment Variables (set BEFORE any import)

```python
import os

# Prevents HuggingFace from sending HEAD requests to huggingface.co on every
# model load. Without this, from_pretrained() retries 5× with exponential
# backoff (~30s wasted) on every run, even when everything is local.
# Corporate networks with SSL inspection make this fail loudly.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# Required for Apple Silicon (MPS). VibeVoice's diffusion scheduler contains
# ops that are not yet implemented in MPS kernels. Without this env var,
# those ops crash with NotImplementedError instead of falling back to CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# NOW import torch, transformers, vibevoice
import torch
```

**Why order matters:** both env vars must be set before `import torch` and
`import transformers` are executed. Setting them after has no effect.

---

## FIX 2 — Device Detection

```python
DEVICE = (
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else
    "cpu"
)
```

---

## FIX 3 — Model Loading (MPS-safe)

```python
from vibevoice import (
    VibeVoiceStreamingForConditionalGenerationInference,
    VibeVoiceStreamingProcessor,
)

processor = VibeVoiceStreamingProcessor.from_pretrained(MODEL_PATH)

if DEVICE == "cuda":
    model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",  # or "sdpa" if flash-attn not available
    )

elif DEVICE == "mps":
    # ✅ CORRECT pattern for MPS:
    #   - torch.float32  (bfloat16 not supported on MPS)
    #   - attn_implementation="sdpa"  (flash_attention_2 is CUDA-only)
    #   - device_map=None  (do NOT pass "mps" — load to CPU first, then .to("mps"))
    model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,
        attn_implementation="sdpa",
        device_map=None,
    )
    model.to("mps")

else:  # cpu
    model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,
        device_map="cpu",
        attn_implementation="sdpa",
    )

model.eval()
```

**Verify all parameters landed on the right device:**
```python
param_devices = set(str(p.device) for p in model.parameters())
assert param_devices == {"mps:0"}, f"Expected mps:0, got {param_devices}"
```

---

## FIX 4 — Noise Scheduler Configuration (DO NOT move to GPU)

```python
# Configure DPM-Solver++ — must be called after from_pretrained()
model.model.noise_scheduler = model.model.noise_scheduler.from_config(
    model.model.noise_scheduler.config,
    algorithm_type="sde-dpmsolver++",
    beta_schedule="squaredcos_cap_v2",
)

model.set_ddpm_inference_steps(num_steps=30)  # 30 steps = good quality/speed tradeoff
```

**CRITICAL — Do NOT move the scheduler to MPS/CUDA after this.**

The scheduler's `set_timesteps()` method uses `numpy` extensively on its internal
tensors (`alphas_cumprod`, `lambda_t`, `sigmas`). It also explicitly forces
`self.sigmas = self.sigmas.to("cpu")` at the end. Moving these tensors to MPS
breaks all numpy operations with:
```
TypeError: can't convert mps:0 device type tensor to numpy.
Use Tensor.cpu() to copy the tensor to host memory first.
```

The device mismatch between CPU scheduler tensors and MPS model tensors is
handled by patching `dpm_solver.py` (FIX 6 below).

---

## FIX 5 — Voice Preset Loading

```python
# ✅ CORRECT — use map_location, nothing else
def load_voice_preset(pt_path: str, device: str) -> object:
    return torch.load(pt_path, map_location=torch.device(device), weights_only=False)
```

**CRITICAL — Do NOT write a recursive tensor-mover and apply it to the preset.**

The preset dict has this structure:
```python
{
    "lm":      ModelOutput,   # has .past_key_values attribute
    "tts_lm":  ModelOutput,
    "neg_lm":  ModelOutput,
    "neg_tts_lm": ModelOutput,
    ...
}
```

`ModelOutput` (from HuggingFace transformers) is a subclass of `OrderedDict`.
A naive recursive helper like:
```python
# ❌ WRONG — destroys ModelOutput objects
def move_to_device(obj, device):
    if isinstance(obj, dict):   # ModelOutput matches this!
        return {k: move_to_device(v, device) for k, v in obj.items()}
    ...
```
matches `isinstance(obj, dict)` and silently converts `ModelOutput` → plain
`dict`. This strips `.past_key_values` attribute access. `model.generate()`
then crashes with:
```
AttributeError: 'dict' object has no attribute 'past_key_values'
```

`torch.load(map_location=device)` already places every tensor on the correct
device. No post-processing is needed.

---

## FIX 6 — Patch `dpm_solver.py` (2 edits in the VibeVoice source)

These patches are in:
`VibeVoice/vibevoice/schedule/dpm_solver.py`

### Patch 6a — `set_timesteps()` line ~355

Find:
```python
last_timestep = ((self.config.num_train_timesteps - clipped_idx).numpy()).item()
```
Replace with:
```python
last_timestep = ((self.config.num_train_timesteps - clipped_idx).cpu().numpy()).item()
```

**Why:** `clipped_idx = torch.searchsorted(torch.flip(self.lambda_t, [0]), ...)`.
If `self.lambda_t` ever ends up on MPS (e.g. from a device move), `clipped_idx`
is an MPS tensor. `.numpy()` on an MPS tensor crashes. `.cpu()` first is
defensive and costs nothing for a scalar.

---

### Patch 6b — `step()` method — sigmas device bridge

The scheduler's `self.sigmas` tensor lives on CPU (by design — `set_timesteps`
forces `self.sigmas = self.sigmas.to("cpu")`). But `dpm_solver_first_order_update`
and the second/third-order variants do arithmetic like:
```python
x_t = (sigma_t / sigma_s * torch.exp(-h)) * sample   # CPU scalar * MPS tensor → crash
```

**Fix:** at the top of `step()`, temporarily move `self.sigmas` to the device
of `model_output`, run the computation, then restore:

Find the block right before `model_output = self.convert_model_output(...)`:
```python
        lower_order_second = (
            (self.step_index == len(self.timesteps) - 2) and self.config.lower_order_final and len(self.timesteps) < 15
        )

        model_output = self.convert_model_output(model_output, sample=sample)
```

Replace with:
```python
        lower_order_second = (
            (self.step_index == len(self.timesteps) - 2) and self.config.lower_order_final and len(self.timesteps) < 15
        )

        # Device fix: self.sigmas is kept on CPU by set_timesteps (numpy compat),
        # but update functions multiply it against model_output/sample (MPS/CUDA).
        # Temporarily move sigmas to match the compute device.
        _compute_device = model_output.device
        _orig_sigmas = self.sigmas
        if str(_compute_device) != str(self.sigmas.device):
            self.sigmas = self.sigmas.to(_compute_device)

        model_output = self.convert_model_output(model_output, sample=sample)
```

Then find:
```python
        # Cast sample back to expected dtype
        prev_sample = prev_sample.to(model_output.dtype)

        # upon completion increase step index by one
        self._step_index += 1
```

Replace with:
```python
        # Cast sample back to expected dtype
        prev_sample = prev_sample.to(model_output.dtype)

        # Restore sigmas to CPU (must stay CPU-resident for numpy compat in set_timesteps)
        self.sigmas = _orig_sigmas

        # upon completion increase step index by one
        self._step_index += 1
```

---

## FIX 7 — AudioStreamer Import (path varies by version)

```python
import importlib

AudioStreamer = None
for _mod_path in ("vibevoice.schedule", "vibevoice.modular", "vibevoice"):
    try:
        _m = importlib.import_module(_mod_path)
        if hasattr(_m, "AudioStreamer"):
            AudioStreamer = _m.AudioStreamer
            break
    except ImportError:
        continue

if AudioStreamer is None:
    raise RuntimeError("AudioStreamer not found — ensure VibeVoice[streamingtts] is installed")
```

---

## Synthesis Pattern

```python
import copy, threading

def synthesize(model, processor, text: str, preset: object, cfg_scale: float = 1.5):
    """
    Synthesise text conditioned on a voice preset.
    Returns (audio_float32_ndarray, ttfa_seconds).
    """
    processed = processor.process_input_with_cached_prompt(
        text=text.strip(),
        cached_prompt=preset,          # ← special method, NOT processor.__call__()
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    inputs = {
        k: v.to(DEVICE) if hasattr(v, "to") else v
        for k, v in processed.items()
    }

    audio_streamer = AudioStreamer(batch_size=1, stop_signal=None, timeout=None)
    errors, ttfa = [], []
    t_start = time.perf_counter()

    def _run():
        try:
            model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=cfg_scale,
                tokenizer=processor.tokenizer,
                generation_config={"do_sample": False},
                audio_streamer=audio_streamer,
                all_prefilled_outputs=copy.deepcopy(preset),  # ← deepcopy to avoid state mutation
                verbose=False,
            )
        except Exception as e:
            errors.append(e)
        finally:
            audio_streamer.end()

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    chunks = []
    for chunk in audio_streamer.get_stream(0):
        if not ttfa:
            ttfa.append(time.perf_counter() - t_start)
        arr = chunk.detach().cpu().float().numpy() if torch.is_tensor(chunk) else np.asarray(chunk, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr.reshape(-1)
        chunks.append(arr)

    t.join()
    if errors:
        raise errors[0]

    audio = np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
    return audio, ttfa[0] if ttfa else 0.0
```

---

## MODEL SELECTION — Critical Constraint

Voice presets in `demo/voices/streaming_model/*.pt` are **model-specific**.
They encode pre-computed KV-cache tensors whose head dimension must match the model.

| Model | KV head dim | Compatible presets |
|---|---|---|
| `VibeVoice-Realtime-0.5B` | 64 | `demo/voices/streaming_model/*.pt` |
| `VibeVoice-1.5B` | 128 | requires separate 1.5B presets |

Mixing them crashes with:
```
RuntimeError: Expected size 64 but got size 128 at ...
```

Always use `VibeVoice-Realtime-0.5B` with the bundled presets.

---

## Available Voices (25 total, all in `demo/voices/streaming_model/`)

```
en-Carter_man    en-Frank_man     en-Mike_man      en-Davis_man
en-Grace_woman   en-Emma_woman    de-Spk0_man      de-Spk1_woman
fr-Spk0_man      fr-Spk1_woman    sp-Spk0_woman    sp-Spk1_man
it-Spk0_woman    it-Spk1_man      nl-Spk0_man      nl-Spk1_woman
pt-Spk0_woman    pt-Spk1_man      pl-Spk0_man      pl-Spk1_woman
kr-Spk0_woman    kr-Spk1_man      jp-Spk0_man      jp-Spk1_woman
in-Samuel_man
```

---

## Save Audio

```python
import scipy.io.wavfile as wavfile
import numpy as np

SAMPLE_RATE = 24_000  # VibeVoice always outputs 24 kHz

def save_wav(audio: np.ndarray, path: str):
    pcm = np.clip(audio, -1.0, 1.0)
    pcm_int16 = (pcm * 32767.0).astype(np.int16)  # use 32767, not 32768 (avoids overflow)
    wavfile.write(path, SAMPLE_RATE, pcm_int16)
```

---

## Summary of Errors and Exact Fixes

| Error message | Cause | Fix |
|---|---|---|
| `HuggingFace.co MaxRetryError / SSLCertVerificationError` | `from_pretrained()` checks remote on every call | `os.environ["HF_HUB_OFFLINE"] = "1"` before imports |
| `AttributeError: 'dict' object has no attribute 'past_key_values'` | Recursive tensor mover converted `ModelOutput` → `dict` | Use `torch.load(map_location=device)` only — no post-processing |
| `TypeError: can't convert mps:0 tensor to numpy` in `set_timesteps` | Scheduler tensors moved to MPS, numpy can't read MPS memory | Keep scheduler on CPU; add `.cpu()` before `.numpy()` in `dpm_solver.py` line ~355 |
| `TypeError: can't convert mps:0 tensor to numpy` in `step()` | CPU `self.sigmas` × MPS `sample`/`model_output` device mismatch | Patch `dpm_solver.py` `step()`: move sigmas to compute device before update, restore after |
| `RuntimeError: Expected size 64 but got size 128` | 1.5B voice presets used with 0.5B model (or vice versa) | Always match model size to preset set |
| `ModuleNotFoundError: flash_attn` | flash-attn is CUDA-only, fails on macOS | Remove flash-attn; use `attn_implementation="sdpa"` |
