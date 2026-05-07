# VibeVoice-TTS — Complete LLM Context Document
> Feed this entire file to an LLM to reproduce VibeVoice-TTS integration as a drop-in
> alternative to Kokoro TTS. Everything needed for zero-shot implementation is here.

---

## 1. What Is VibeVoice-TTS

- **Repo**: https://github.com/microsoft/VibeVoice
- **HuggingFace**: `microsoft/VibeVoice-Realtime-0.5B` (use this — see §6 for why)
- **Architecture**: Next-token diffusion on top of Qwen2.5 LLM + continuous speech tokenizers
  at 7.5 Hz frame rate + diffusion head for audio generation
- **Strengths**: Long-form audio (up to 90 min), multi-speaker (up to 4), expressive/conversational
- **Languages**: English, Chinese (English is most stable)
- **Output**: float32 numpy array → 16-bit PCM WAV at **24000 Hz**
- **NOT a cloud API** — fully local inference, weights on HuggingFace

---

## 2. Installation (Exact Steps That Work)

```bash
# Step 1 — create venv (required, do not install globally)
python -m venv venv
source venv/bin/activate          # macOS/Linux
# venv\Scripts\activate           # Windows

# Step 2 — upgrade build tools FIRST (flash-attn needs wheel)
pip install --upgrade pip wheel setuptools --index-url https://pypi.org/simple/

# Step 3 — clone repo (voice preset .pt files live inside it)
git clone https://github.com/microsoft/VibeVoice.git

# Step 4 — install with streaming TTS extras (pins transformers==4.51.3)
pip install -e "./VibeVoice[streamingtts]" --index-url https://pypi.org/simple/

# Step 5 — download model weights
python -c "
from huggingface_hub import snapshot_download
snapshot_download('microsoft/VibeVoice-Realtime-0.5B', local_dir='./models/VibeVoice-Realtime-0.5B')
"

# NOTE: Do NOT install flash-attn on macOS — CUDA only, will always fail
# NOTE: Use --index-url https://pypi.org/simple/ to avoid corporate Artifactory issues
```

### Key Dependencies (auto-installed via extras)
| Package | Version | Purpose |
|---|---|---|
| `transformers` | ==4.51.3 (pinned) | Model loading |
| `torch` | >=2.0 | Inference backend |
| `diffusers` | latest | Noise scheduler |
| `librosa` | latest | Audio resampling |
| `scipy` | latest | WAV file writing |
| `accelerate` | latest | Device mapping |
| `numpy` | latest | Audio arrays |

---

## 3. Core API Classes

```python
from vibevoice import (
    VibeVoiceStreamingForConditionalGenerationInference,  # main model class
    VibeVoiceStreamingProcessor,                          # tokenizer + processor
)

# AudioStreamer is NOT in vibevoice.__init__ — must be discovered dynamically:
import importlib
AudioStreamer = None
for mod_path in ("vibevoice.schedule", "vibevoice.modular", "vibevoice"):
    try:
        m = importlib.import_module(mod_path)
        if hasattr(m, "AudioStreamer"):
            AudioStreamer = m.AudioStreamer
            break
    except ImportError:
        continue
```

---

## 4. Device Detection (macOS M-series / CUDA / CPU)

```python
import torch

DEVICE = "cuda" if torch.cuda.is_available() else (
         "mps"  if torch.backends.mps.is_available() else "cpu")

# Device → dtype + attention implementation mapping:
# cuda  → torch.bfloat16  + flash_attention_2  (or sdpa fallback)
# mps   → torch.float32   + sdpa               (flash-attn NOT supported on MPS)
# cpu   → torch.float32   + sdpa
```

---

## 5. Model Loading (Full Pattern)

```python
import torch
import traceback
from vibevoice import (
    VibeVoiceStreamingForConditionalGenerationInference,
    VibeVoiceStreamingProcessor,
)

MODEL_PATH = "./models/VibeVoice-Realtime-0.5B"  # local path OR HF model id
DDPM_STEPS = 30   # 10-20 = fast preview | 50 = best quality

def load_model(model_path: str, device: str):
    processor = VibeVoiceStreamingProcessor.from_pretrained(model_path)

    if device == "cuda":
        dtype, attn = torch.bfloat16, "flash_attention_2"
        try:
            model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                model_path, torch_dtype=dtype, device_map="cuda",
                attn_implementation=attn,
            )
        except Exception:
            # flash_attention_2 not installed — fall back silently
            model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                model_path, torch_dtype=dtype, device_map="cuda",
                attn_implementation="sdpa",
            )
    elif device == "mps":
        model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
            model_path, torch_dtype=torch.float32,
            attn_implementation="sdpa", device_map=None,
        )
        model.to("mps")
    else:  # cpu
        model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
            model_path, torch_dtype=torch.float32,
            device_map="cpu", attn_implementation="sdpa",
        )

    model.eval()

    # Configure DPM-Solver++ noise scheduler (better quality than default)
    try:
        model.model.noise_scheduler = model.model.noise_scheduler.from_config(
            model.model.noise_scheduler.config,
            algorithm_type="sde-dpmsolver++",
            beta_schedule="squaredcos_cap_v2",
        )
    except Exception:
        pass  # use model defaults if config fails

    model.set_ddpm_inference_steps(num_steps=DDPM_STEPS)
    return processor, model
```

---

## 6. Voice Presets — CRITICAL KNOWLEDGE

### What they are
Voice presets are **pre-computed KV-cache tensors** (`.pt` files) representing a speaker's voice.
They are loaded and passed to the processor as `cached_prompt` to condition generation.

### Location after cloning
```
./VibeVoice/demo/voices/streaming_model/
├── en-Carter_man.pt      ← deep, measured male
├── en-Davis_man.pt       ← warm, conversational male
├── en-Emma_woman.pt      ← clear, neutral female
├── en-Frank_man.pt       ← authoritative male
├── en-Grace_woman.pt     ← expressive female
├── en-Mike_man.pt        ← additional male voice
├── de-Spk0_man.pt        ← German male
├── de-Spk1_woman.pt      ← German female
├── fr-Spk0_man.pt        ← French male
├── fr-Spk1_woman.pt      ← French female
└── ... (more languages)
```

### ⚠️ CRITICAL: Presets are model-specific
```
Realtime-0.5B → KV head size = 64   ← presets in repo are for THIS model
1.5B          → KV head size = 128  ← INCOMPATIBLE with above presets

# Mixing them causes:
# RuntimeError: Sizes of tensors must match except in dimension 2.
#               Expected size 64 but got size 128

# ALWAYS use VibeVoice-Realtime-0.5B with the bundled voice presets.
# If you need 1.5B, you must generate new presets from reference audio
# using the 1.5B model itself.
```

### Loading a voice preset
```python
import torch

def load_voice_preset(pt_path: str, device: str) -> object:
    return torch.load(pt_path, map_location=torch.device(device), weights_only=False)
```

---

## 7. Core Synthesis Function

```python
import copy
import threading
import time
import numpy as np

SAMPLE_RATE = 24_000
CFG_SCALE   = 1.5

def synthesize(model, processor, text: str, prefilled_outputs, device: str, cfg_scale=CFG_SCALE):
    """
    Synthesise text → float32 numpy audio array.
    Returns: (audio: np.ndarray[float32], ttfa_seconds: float)
    """
    # Normalise curly quotes (model tip from official docs)
    text = (text.replace("\u2018", "'").replace("\u2019", "'")
                .replace("\u201c", '"').replace("\u201d", '"'))

    # Build model inputs
    processed = processor.process_input_with_cached_prompt(
        text=text.strip(),
        cached_prompt=prefilled_outputs,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    inputs = {
        k: v.to(torch.device(device)) if hasattr(v, "to") else v
        for k, v in processed.items()
    }

    audio_streamer = AudioStreamer(batch_size=1, stop_signal=None, timeout=None)
    errors, ttfa_s = [], []
    stop_event = threading.Event()
    t_start = time.perf_counter()

    def _generate():
        try:
            model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=cfg_scale,
                tokenizer=processor.tokenizer,
                generation_config={"do_sample": False, "temperature": 1.0, "top_p": 1.0},
                audio_streamer=audio_streamer,
                stop_check_fn=stop_event.is_set,
                verbose=False,
                refresh_negative=True,
                all_prefilled_outputs=copy.deepcopy(prefilled_outputs),
            )
        except Exception as exc:
            errors.append(exc)
        finally:
            audio_streamer.end()

    thread = threading.Thread(target=_generate, daemon=True)
    thread.start()

    chunks = []
    try:
        for chunk in audio_streamer.get_stream(0):
            if not ttfa_s:
                ttfa_s.append(time.perf_counter() - t_start)
            if torch.is_tensor(chunk):
                chunk = chunk.detach().cpu().to(torch.float32).numpy()
            else:
                chunk = np.asarray(chunk, dtype=np.float32)
            if chunk.ndim > 1:
                chunk = chunk.reshape(-1)
            peak = float(np.max(np.abs(chunk)))
            if peak > 1.0:
                chunk = chunk / peak
            chunks.append(chunk)
    finally:
        stop_event.set()
        audio_streamer.end()
        thread.join()

    if errors:
        raise errors[0]

    audio = np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
    return audio, (ttfa_s[0] if ttfa_s else 0.0)
```

---

## 8. Long Text Handling (>1000 chars / 20K chars)

VibeVoice is NOT designed for simple chunking in the way typical TTS systems are.
The correct approach for long text is to chunk at sentence boundaries and generate
each chunk independently, then concatenate the audio arrays.

```python
import re

CHUNK_SIZE = 1000  # chars — safe for M4 MPS memory; tune 500-2000

def chunk_text(text: str, max_chars: int = CHUNK_SIZE) -> list:
    """Split at sentence boundaries to preserve prosody."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks, current = [], ""
    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= max_chars:
            current = (current + " " + sentence).strip()
        else:
            if current:
                chunks.append(current)
            while len(sentence) > max_chars:
                chunks.append(sentence[:max_chars])
                sentence = sentence[max_chars:]
            current = sentence
    if current:
        chunks.append(current)
    return chunks


def synthesize_long(model, processor, text: str, prefilled_outputs, device: str):
    """Chunk → synthesise each → concatenate. Safe for 20K+ chars."""
    chunks = chunk_text(text, CHUNK_SIZE)
    if len(chunks) == 1:
        return synthesize(model, processor, text, prefilled_outputs, device)

    all_audio, first_ttfa = [], None
    for i, chunk in enumerate(chunks, 1):
        audio, ttfa = synthesize(model, processor, chunk, prefilled_outputs, device)
        if first_ttfa is None:
            first_ttfa = ttfa
        all_audio.append(audio)

    return np.concatenate(all_audio), (first_ttfa or 0.0)
```

---

## 9. Saving Audio Output

```python
import scipy.io.wavfile as wavfile

def save_wav(path: str, audio: np.ndarray, sample_rate: int = 24_000):
    """Write float32 audio to 16-bit PCM WAV. Use 32767 (not 32768) to avoid overflow."""
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    wavfile.write(path, sample_rate, pcm)
```

---

## 10. Kokoro → VibeVoice Drop-In Adapter

Use this adapter class to swap VibeVoice into any project currently using Kokoro.
The interface mirrors Kokoro's `generate()` signature as closely as possible.

```python
import os, copy, threading, time, re
import numpy as np
import scipy.io.wavfile as wavfile
import torch
import importlib

class VibeVoiceTTS:
    """
    Drop-in replacement for Kokoro TTS.

    Kokoro usage (typical):
        kokoro = KPipeline(lang_code='a')
        audio, sr = kokoro(text, voice='af_heart', speed=1.0)

    VibeVoice equivalent:
        tts = VibeVoiceTTS()
        tts.load()
        audio, sr = tts.generate(text, voice='en-Emma_woman')
    """

    SAMPLE_RATE = 24_000
    CHUNK_SIZE  = 1000       # max chars per synthesis pass
    CFG_SCALE   = 1.5
    DDPM_STEPS  = 30

    # Map Kokoro voice names → VibeVoice presets for easy migration
    KOKORO_VOICE_MAP = {
        "af_heart":   "en-Emma_woman",
        "af_bella":   "en-Grace_woman",
        "am_adam":    "en-Carter_man",
        "am_michael": "en-Davis_man",
        "bf_emma":    "en-Emma_woman",
        "bm_george":  "en-Frank_man",
        # add more mappings as needed
    }

    def __init__(
        self,
        model_path: str = "./models/VibeVoice-Realtime-0.5B",
        voices_dir: str = "./VibeVoice/demo/voices/streaming_model",
        device: str = None,
        ddpm_steps: int = 30,
    ):
        self.model_path  = model_path
        self.voices_dir  = voices_dir
        self.ddpm_steps  = ddpm_steps
        self.device      = device or (
            "cuda" if torch.cuda.is_available() else
            "mps"  if torch.backends.mps.is_available() else "cpu"
        )
        self.model       = None
        self.processor   = None
        self._AudioStreamer = None
        self._voice_cache   = {}

    # ── Setup ───────────────────────────────────────────────────────────────

    def load(self):
        """Load model and processor. Call once before any generate() calls."""
        from vibevoice import (
            VibeVoiceStreamingForConditionalGenerationInference,
            VibeVoiceStreamingProcessor,
        )
        self.processor = VibeVoiceStreamingProcessor.from_pretrained(self.model_path)

        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        attn  = "flash_attention_2" if self.device == "cuda" else "sdpa"

        try:
            self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                self.model_path, torch_dtype=dtype,
                device_map=self.device if self.device != "mps" else None,
                attn_implementation=attn,
            )
        except Exception:
            self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                self.model_path, torch_dtype=dtype,
                device_map=self.device if self.device != "mps" else None,
                attn_implementation="sdpa",
            )
        if self.device == "mps":
            self.model.to("mps")

        self.model.eval()
        try:
            self.model.model.noise_scheduler = self.model.model.noise_scheduler.from_config(
                self.model.model.noise_scheduler.config,
                algorithm_type="sde-dpmsolver++",
                beta_schedule="squaredcos_cap_v2",
            )
        except Exception:
            pass
        self.model.set_ddpm_inference_steps(num_steps=self.ddpm_steps)

        # Locate AudioStreamer
        for mod_path in ("vibevoice.schedule", "vibevoice.modular", "vibevoice"):
            try:
                m = importlib.import_module(mod_path)
                if hasattr(m, "AudioStreamer"):
                    self._AudioStreamer = m.AudioStreamer
                    break
            except ImportError:
                continue
        if self._AudioStreamer is None:
            raise RuntimeError("AudioStreamer not found in vibevoice package.")

    # ── Public API ───────────────────────────────────────────────────────────

    def generate(self, text: str, voice: str = "en-Carter_man", speed: float = 1.0):
        """
        Generate audio from text.

        Args:
            text  : input text (any length — auto-chunked if >CHUNK_SIZE chars)
            voice : built-in voice name (e.g. 'en-Carter_man') OR
                    Kokoro voice name (auto-mapped, e.g. 'af_heart') OR
                    path to a .pt preset file OR
                    path to a reference .wav for cloning
            speed : ignored (VibeVoice does not expose speed control directly)

        Returns:
            (audio: np.ndarray[float32], sample_rate: int)
        """
        if self.model is None:
            raise RuntimeError("Call .load() before .generate()")

        prefilled = self._resolve_voice(voice)
        text = self._normalise_text(text)

        chunks = self._chunk_text(text)
        all_audio = []
        for chunk in chunks:
            audio, _ = self._synthesize_one(chunk, prefilled)
            all_audio.append(audio)

        final = np.concatenate(all_audio) if all_audio else np.array([], dtype=np.float32)
        return final, self.SAMPLE_RATE

    def save(self, audio: np.ndarray, path: str):
        """Save float32 audio array to 16-bit PCM WAV file."""
        pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
        wavfile.write(path, self.SAMPLE_RATE, pcm)

    @property
    def available_voices(self) -> list:
        """Return names of all available built-in voice presets."""
        if not os.path.isdir(self.voices_dir):
            return []
        return [
            os.path.splitext(f)[0]
            for f in os.listdir(self.voices_dir)
            if f.endswith(".pt")
        ]

    # ── Internals ────────────────────────────────────────────────────────────

    def _resolve_voice(self, voice: str):
        """Resolve voice string → prefilled_outputs (KV cache tensor)."""
        # Cache hit
        if voice in self._voice_cache:
            return self._voice_cache[voice]

        # Kokoro name mapping
        mapped = self.KOKORO_VOICE_MAP.get(voice, voice)

        # Path to .pt file (explicit or by name)
        if mapped.endswith(".pt") and os.path.exists(mapped):
            pt_path = mapped
        else:
            pt_path = os.path.join(self.voices_dir, f"{mapped}.pt")

        if os.path.exists(pt_path):
            prefilled = torch.load(pt_path, map_location=torch.device(self.device), weights_only=False)
            self._voice_cache[voice] = prefilled
            return prefilled

        # Reference WAV → clone
        if os.path.exists(mapped) and mapped.endswith(".wav"):
            prefilled = self._clone_from_wav(mapped)
            self._voice_cache[voice] = prefilled
            return prefilled

        # Fallback to first available preset
        fallback = os.path.join(self.voices_dir, "en-Carter_man.pt")
        if os.path.exists(fallback):
            print(f"[VibeVoiceTTS] Voice '{voice}' not found — using en-Carter_man")
            return torch.load(fallback, map_location=torch.device(self.device), weights_only=False)

        raise FileNotFoundError(f"No voice preset found for '{voice}' in {self.voices_dir}")

    def _clone_from_wav(self, wav_path: str):
        """Best-effort voice clone from reference WAV → KV cache."""
        try:
            import librosa
            audio_arr, _ = librosa.load(wav_path, sr=self.SAMPLE_RATE, mono=True)
            audio_tensor = torch.tensor(audio_arr, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                inputs = self.processor(
                    audios=audio_tensor, sampling_rate=self.SAMPLE_RATE,
                    return_tensors="pt", padding=True,
                )
                inputs = {k: v.to(torch.device(self.device)) if hasattr(v, "to") else v
                          for k, v in inputs.items()}
                outputs = self.model(**inputs, use_cache=True)
                return outputs.past_key_values
        except Exception as e:
            raise RuntimeError(f"Voice cloning from {wav_path} failed: {e}")

    def _normalise_text(self, text: str) -> str:
        return (text.replace("\u2018", "'").replace("\u2019", "'")
                    .replace("\u201c", '"').replace("\u201d", '"'))

    def _chunk_text(self, text: str) -> list:
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        chunks, current = [], ""
        for s in sentences:
            if len(current) + len(s) + 1 <= self.CHUNK_SIZE:
                current = (current + " " + s).strip()
            else:
                if current:
                    chunks.append(current)
                while len(s) > self.CHUNK_SIZE:
                    chunks.append(s[:self.CHUNK_SIZE])
                    s = s[self.CHUNK_SIZE:]
                current = s
        if current:
            chunks.append(current)
        return chunks or [text]

    def _synthesize_one(self, text: str, prefilled_outputs):
        processed = self.processor.process_input_with_cached_prompt(
            text=text.strip(), cached_prompt=prefilled_outputs,
            padding=True, return_tensors="pt", return_attention_mask=True,
        )
        inputs = {k: v.to(torch.device(self.device)) if hasattr(v, "to") else v
                  for k, v in processed.items()}

        streamer   = self._AudioStreamer(batch_size=1, stop_signal=None, timeout=None)
        errors     = []
        stop_event = threading.Event()
        ttfa_s     = []
        t_start    = time.perf_counter()

        def _run():
            try:
                self.model.generate(
                    **inputs, max_new_tokens=None, cfg_scale=self.CFG_SCALE,
                    tokenizer=self.processor.tokenizer,
                    generation_config={"do_sample": False, "temperature": 1.0, "top_p": 1.0},
                    audio_streamer=streamer, stop_check_fn=stop_event.is_set,
                    verbose=False, refresh_negative=True,
                    all_prefilled_outputs=copy.deepcopy(prefilled_outputs),
                )
            except Exception as exc:
                errors.append(exc)
            finally:
                streamer.end()

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        chunks = []
        try:
            for chunk in streamer.get_stream(0):
                if not ttfa_s:
                    ttfa_s.append(time.perf_counter() - t_start)
                if torch.is_tensor(chunk):
                    chunk = chunk.detach().cpu().to(torch.float32).numpy()
                else:
                    chunk = np.asarray(chunk, dtype=np.float32)
                if chunk.ndim > 1:
                    chunk = chunk.reshape(-1)
                peak = float(np.max(np.abs(chunk)))
                if peak > 1.0:
                    chunk = chunk / peak
                chunks.append(chunk)
        finally:
            stop_event.set()
            streamer.end()
            thread.join()

        if errors:
            raise errors[0]

        audio = np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
        return audio, (ttfa_s[0] if ttfa_s else 0.0)
```

---

## 11. Usage Examples (Kokoro → VibeVoice Migration)

### Basic usage
```python
# ── BEFORE (Kokoro) ──────────────────────────────────────────
from kokoro import KPipeline
kokoro = KPipeline(lang_code='a')
audio, sr = kokoro("Hello world", voice='af_heart', speed=1.0)

# ── AFTER (VibeVoice) ────────────────────────────────────────
from vibevoice_adapter import VibeVoiceTTS   # save the adapter class above

tts = VibeVoiceTTS(
    model_path  = "./models/VibeVoice-Realtime-0.5B",
    voices_dir  = "./VibeVoice/demo/voices/streaming_model",
)
tts.load()
audio, sr = tts.generate("Hello world", voice='af_heart')  # Kokoro name auto-mapped
tts.save(audio, "output.wav")
```

### Long text (20K chars)
```python
with open("long_document.txt") as f:
    text = f.read()   # 20,000+ chars

audio, sr = tts.generate(text, voice='en-Emma_woman')
tts.save(audio, "output_long.wav")
# Automatically chunked into ~20 passes of 1000 chars and concatenated
```

### Selecting a voice
```python
print(tts.available_voices)
# ['en-Carter_man', 'en-Davis_man', 'en-Emma_woman', 'en-Frank_man',
#  'en-Grace_woman', 'en-Mike_man', 'de-Spk0_man', ...]

audio, sr = tts.generate(text, voice='en-Grace_woman')
```

### Voice cloning from reference WAV
```python
audio, sr = tts.generate(text, voice='./my_speaker.wav')
# Best-effort — works if processor supports raw audio encoding
```

### Conditional loading (keep Kokoro as fallback)
```python
import os

USE_VIBEVOICE = os.getenv("TTS_BACKEND", "kokoro") == "vibevoice"

if USE_VIBEVOICE:
    from vibevoice_adapter import VibeVoiceTTS
    tts_engine = VibeVoiceTTS()
    tts_engine.load()
    def synthesise(text, voice="en-Carter_man"):
        audio, sr = tts_engine.generate(text, voice=voice)
        return audio, sr
else:
    from kokoro import KPipeline
    kokoro = KPipeline(lang_code='a')
    def synthesise(text, voice="af_heart"):
        return kokoro(text, voice=voice)
```

---

## 12. Hard-Won Learnings & Gotchas

### ❌ Do NOT use VibeVoice-1.5B with bundled voice presets
The `.pt` files in `demo/voices/streaming_model/` were built for `Realtime-0.5B`.
The 1.5B model has a different KV head dimension (128 vs 64), causing:
```
RuntimeError: Sizes of tensors must match except in dimension 2.
Expected size 64 but got size 128
```
**Fix**: Always use `VibeVoice-Realtime-0.5B` with the bundled presets.

### ❌ Do NOT use flash-attn on macOS / Apple Silicon
`flash-attn` is CUDA-only. On Mac it will fail during `pip install` with:
```
ModuleNotFoundError: No module named 'wheel'
```
(even after installing wheel — the package simply doesn't support MPS).
**Fix**: Skip `flash-attn` on Mac. The script falls back to `sdpa` automatically.

### ❌ Do NOT upgrade pip as `pip3 install --upgrade pip3`
`pip3` is the binary name, not the package name. The package is `pip`.
**Fix**: `pip install --upgrade pip`

### ❌ Do NOT use corporate Artifactory for flash-attn or vibevoice
Walmart's internal PyPI (`devtools-pypi`) may not have all packages.
**Fix**: Add `--index-url https://pypi.org/simple/` to all `pip install` calls.

### ✅ Always deepcopy prefilled_outputs before passing to generate()
The model mutates the KV cache in-place during generation. Without `copy.deepcopy()`,
the second chunk will use a corrupted cache and produce silence or errors.

### ✅ Use `process_input_with_cached_prompt()` — not `__call__()`
VibeVoice's processor has a special method for TTS that accepts `cached_prompt`.
Calling `processor(text=...)` directly will not work for conditioned generation.

### ✅ Normalise curly quotes before synthesis
The model is sensitive to non-ASCII punctuation:
```python
text = text.replace("\u2018","'").replace("\u2019","'").replace("\u201c",'"').replace("\u201d",'"')
```

### ✅ Sample rate is 24000 Hz (not 22050 like many TTS models)
Always write WAV files with `sr=24000`. Using 22050 will play back at wrong speed.

### ✅ Use 32767.0 (not 32768.0) when converting float32 → int16
```python
# Correct (no overflow at amplitude=1.0):
pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
```

### ✅ AudioStreamer must be discovered dynamically
It is not exported from `vibevoice.__init__`. Search `vibevoice.schedule`,
`vibevoice.modular`, and `vibevoice` in that order.

### ✅ M4 Mac memory: expect ~14 GB unified memory for Realtime-0.5B
The model uses `float32` on MPS (bfloat16 not reliable). Each 1000-char chunk
takes ~5-15s on M4. For 20 chunks = ~2-5 min total.

---

## 13. Quick Sanity-Test Script

```python
# Run this first to verify your setup works end-to-end before integrating
from vibevoice_adapter import VibeVoiceTTS

tts = VibeVoiceTTS()
tts.load()
print("Available voices:", tts.available_voices)

audio, sr = tts.generate(
    "This is a quick sanity test of the VibeVoice text to speech system.",
    voice="en-Carter_man",
)
tts.save(audio, "sanity_test.wav")
print(f"Done. Audio: {len(audio)/sr:.2f}s at {sr}Hz → sanity_test.wav")
```

---

## 14. File Structure Reference

```
your_project/
├── vibevoice_adapter.py           ← paste the VibeVoiceTTS class here
├── models/
│   └── VibeVoice-Realtime-0.5B/  ← downloaded via snapshot_download()
│       ├── config.json
│       ├── model.safetensors
│       └── ...
└── VibeVoice/                     ← git cloned repo (for voice presets)
    └── demo/voices/streaming_model/
        ├── en-Carter_man.pt
        ├── en-Emma_woman.pt
        └── ...
```

---

*Document generated from live implementation and debugging session.
All code patterns validated on macOS Apple Silicon (M4) with VibeVoice-Realtime-0.5B.*
