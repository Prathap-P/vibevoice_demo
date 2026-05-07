#!/usr/bin/env python3
"""
vibevoice_eval.py
──────────────────────────────────────────────────────────────────────────────
VibeVoice-TTS  |  Quick-Look Evaluation Script
  • MODE 1 – Standard : synthesises input_text.txt with the top-5 built-in
              English voices; outputs output_{voice_name}.wav + metrics.
  • MODE 2 – Cloning  : zero-shot voice clone from reference.wav (best-effort);
              outputs output_cloned.wav.
──────────────────────────────────────────────────────────────────────────────
"""

import copy
import os
import threading
import time
import traceback

import numpy as np
import scipy.io.wavfile as wavfile
import torch

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
MODEL_PATH   = "./models/VibeVoice-Realtime-0.5B"  # voice presets are built for this model — do NOT change to 1.5B
VOICES_DIR   = "./VibeVoice/demo/voices/streaming_model"
INPUT_TXT    = "input_text.txt"
REF_AUDIO    = "reference.wav"
SAMPLE_RATE  = 24_000
DEVICE       = "cuda" if torch.cuda.is_available() else (
               "mps"  if torch.backends.mps.is_available() else "cpu")
DDPM_STEPS   = 30
CFG_SCALE    = 1.5
CHUNK_SIZE   = 1000   # max chars per synthesis chunk (tweak 500–2000 to balance quality vs memory)

TOP_5_VOICES = [
    "en-Carter_man",
    "en-Emma_woman",
    "en-Davis_man",
    "en-Grace_woman",
    "en-Frank_man",
]

# ──────────────────────────────────────────────────────────────────────────────
# PACKAGE IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
try:
    from vibevoice import (
        VibeVoiceStreamingForConditionalGenerationInference,
        VibeVoiceStreamingProcessor,
    )
except ImportError as exc:
    print(f"[FATAL] vibevoice not installed: {exc}")
    print("  → Run:  pip install -e './VibeVoice[streamingtts]'")
    raise SystemExit(1)

# AudioStreamer — try multiple submodule paths for robustness
AudioStreamer = None
for _mod_path in ("vibevoice.schedule", "vibevoice.modular", "vibevoice"):
    try:
        import importlib
        _m = importlib.import_module(_mod_path)
        if hasattr(_m, "AudioStreamer"):
            AudioStreamer = _m.AudioStreamer
            break
    except ImportError:
        continue

if AudioStreamer is None:
    print("[FATAL] Could not locate AudioStreamer. Ensure the repo is installed correctly.")
    raise SystemExit(1)


# ──────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ──────────────────────────────────────────────────────────────────────────────
def load_model(model_path: str, device: str) -> tuple:
    """Load processor + model with optimal device settings."""
    print(f"\n[LOAD] Processor  ← {model_path}")
    processor = VibeVoiceStreamingProcessor.from_pretrained(model_path)

    print(f"[LOAD] Model      ← {model_path}  (device={device})")
    t0 = time.perf_counter()

    if device == "cuda":
        dtype, attn = torch.bfloat16, "flash_attention_2"
        try:
            model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map="cuda",
                attn_implementation=attn,
            )
        except Exception:
            print("  flash_attention_2 unavailable → falling back to sdpa")
            model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map="cuda",
                attn_implementation="sdpa",
            )
    elif device == "mps":
        model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            attn_implementation="sdpa",
            device_map=None,
        )
        model.to("mps")
    else:
        model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map="cpu",
            attn_implementation="sdpa",
        )

    model.eval()

    try:
        model.model.noise_scheduler = model.model.noise_scheduler.from_config(
            model.model.noise_scheduler.config,
            algorithm_type="sde-dpmsolver++",
            beta_schedule="squaredcos_cap_v2",
        )
    except Exception as e:
        print(f"  [WARN] Noise-scheduler tweak skipped ({e}); using defaults.")
    model.set_ddpm_inference_steps(num_steps=DDPM_STEPS)

    print(f"  ✅  Model ready in {time.perf_counter() - t0:.1f}s")
    return processor, model


# ──────────────────────────────────────────────────────────────────────────────
# VOICE PRESET LOADING
# ──────────────────────────────────────────────────────────────────────────────
def load_voice_preset(pt_path: str) -> object:
    """Load a pre-computed KV-cache voice preset from a .pt file."""
    return torch.load(
        pt_path,
        map_location=torch.device(DEVICE),
        weights_only=False,
    )


def build_voice_preset_from_wav(model, processor, wav_path: str) -> object:
    """
    Build a KV-cache voice preset from a reference WAV file.
    Tries three strategies in order:
      A. processor helper method
      B. encode audio → run model forward → extract past_key_values
      C. fall back to first available built-in .pt preset
    """
    # Strategy A: processor helper
    for fn_name in (
        "create_voice_preset",
        "encode_reference_audio",
        "build_cached_prompt",
        "process_reference_audio",
    ):
        fn = getattr(processor, fn_name, None)
        if callable(fn):
            print(f"  → processor.{fn_name}() found — using it for cloning.")
            return fn(wav_path, return_tensors="pt")

    # Strategy B: manual KV-cache extraction
    try:
        import librosa
        audio_arr, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
        audio_tensor = torch.tensor(audio_arr, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            clone_inputs = processor(
                audios=audio_tensor,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
                padding=True,
            )
            clone_inputs = {
                k: v.to(torch.device(DEVICE)) if hasattr(v, "to") else v
                for k, v in clone_inputs.items()
            }
            outputs = model(**clone_inputs, use_cache=True)
            print("  → KV-cache extracted from reference audio.")
            return outputs.past_key_values
    except Exception as e:
        print(f"  [WARN] KV-cache extraction failed ({type(e).__name__}: {e})")

    # Strategy C: built-in fallback
    fallback = os.path.join(VOICES_DIR, f"{TOP_5_VOICES[0]}.pt")
    if os.path.exists(fallback):
        print(f"  [FALLBACK] Using built-in voice: {TOP_5_VOICES[0]}")
        return load_voice_preset(fallback)

    raise RuntimeError("Cloning failed: reference.wav unprocessable and no built-in .pt found.")


# ──────────────────────────────────────────────────────────────────────────────
# CORE SYNTHESIS
# ──────────────────────────────────────────────────────────────────────────────
def synthesize(
    model,
    processor,
    text: str,
    prefilled_outputs: object,
    cfg_scale: float = CFG_SCALE,
) -> tuple:
    """
    Synthesise text conditioned on prefilled_outputs (voice KV cache).
    Returns: (audio: float32 ndarray, ttfa_s: float)
    """
    processed = processor.process_input_with_cached_prompt(
        text=text.strip(),
        cached_prompt=prefilled_outputs,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    inputs = {
        k: v.to(torch.device(DEVICE)) if hasattr(v, "to") else v
        for k, v in processed.items()
    }

    audio_streamer = AudioStreamer(batch_size=1, stop_signal=None, timeout=None)
    errors: list = []
    stop_event = threading.Event()
    ttfa_s: list = []
    t_start = time.perf_counter()

    def _generate() -> None:
        try:
            model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=cfg_scale,
                tokenizer=processor.tokenizer,
                generation_config={
                    "do_sample": False,
                    "temperature": 1.0,
                    "top_p": 1.0,
                },
                audio_streamer=audio_streamer,
                stop_check_fn=stop_event.is_set,
                verbose=False,
                refresh_negative=True,
                all_prefilled_outputs=copy.deepcopy(prefilled_outputs),
            )
        except Exception as exc:
            errors.append(exc)
            traceback.print_exc()
        finally:
            audio_streamer.end()

    gen_thread = threading.Thread(target=_generate, daemon=True)
    gen_thread.start()

    chunks: list = []
    try:
        stream = audio_streamer.get_stream(0)
        for chunk in stream:
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
        gen_thread.join()

    if errors:
        raise errors[0]

    audio = np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
    return audio, ttfa_s[0] if ttfa_s else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# TEXT CHUNKING  (safe split at sentence boundaries)
# ──────────────────────────────────────────────────────────────────────────────
def chunk_text(text: str, max_chars: int = CHUNK_SIZE) -> list:
    """
    Split text into chunks of at most max_chars, always breaking at sentence
    boundaries (. ! ?) to preserve natural prosody across chunks.
    """
    import re
    # Split on sentence-ending punctuation, keeping the delimiter
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks, current = [], ""
    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= max_chars:
            current = (current + " " + sentence).strip()
        else:
            if current:
                chunks.append(current)
            # If a single sentence exceeds max_chars, hard-split it
            while len(sentence) > max_chars:
                chunks.append(sentence[:max_chars])
                sentence = sentence[max_chars:]
            current = sentence
    if current:
        chunks.append(current)
    return chunks


def synthesize_long(
    model,
    processor,
    text: str,
    prefilled_outputs: object,
    cfg_scale: float = CFG_SCALE,
) -> tuple:
    """
    Synthesise long text (>CHUNK_SIZE chars) by splitting into chunks,
    generating each independently, and concatenating the audio.
    Returns: (audio: float32 ndarray, ttfa_s: float)
    """
    chunks = chunk_text(text, CHUNK_SIZE)
    total  = len(chunks)

    if total == 1:
        # Short enough — single pass
        return synthesize(model, processor, text, prefilled_outputs, cfg_scale)

    print(f"  📄  Long text detected — split into {total} chunks of ≤{CHUNK_SIZE} chars")
    all_audio, first_ttfa = [], None

    for i, chunk in enumerate(chunks, 1):
        print(f"  ⏳  Chunk {i}/{total}  ({len(chunk)} chars)…")
        audio, ttfa = synthesize(model, processor, chunk, prefilled_outputs, cfg_scale)
        if first_ttfa is None:
            first_ttfa = ttfa
        all_audio.append(audio)

    return np.concatenate(all_audio), first_ttfa or 0.0


# ──────────────────────────────────────────────────────────────────────────────
# OUTPUT UTILITIES
# ──────────────────────────────────────────────────────────────────────────────
def save_wav(out_path: str, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    """Write float32 audio to a 16-bit PCM WAV file."""
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    wavfile.write(out_path, sample_rate, pcm)
    dur = len(audio) / sample_rate
    print(f"  💾  Saved → {out_path}  ({dur:.2f}s  |  {len(audio):,} samples)")


def print_report(label: str, text: str, audio: np.ndarray, total_elapsed: float, ttfa_s: float) -> None:
    """Print a concise performance report."""
    duration = len(audio) / SAMPLE_RATE
    rtf      = total_elapsed / duration if duration > 0 else float("inf")
    words    = len(text.split())
    realtime = "✅ real-time" if rtf <= 1.0 else "⏳ slower-than-real-time"
    sep = "─" * 54
    print(f"\n  {sep}")
    print(f"  📊  [{label}]")
    print(f"  {sep}")
    print(f"  Input text    : {len(text):>6} chars  /  {words} words")
    print(f"  Audio dur.    : {duration:>6.2f}s")
    print(f"  Time-to-first : {ttfa_s:>6.2f}s   (TTFA)")
    print(f"  Total infer.  : {total_elapsed:>6.2f}s")
    print(f"  RTF           : {rtf:>6.3f}x   {realtime}")
    print(f"  {sep}")


# ──────────────────────────────────────────────────────────────────────────────
# MODE 1 — STANDARD
# ──────────────────────────────────────────────────────────────────────────────
def run_standard_mode(processor, model, text: str) -> None:
    print(f"\n{'='*60}")
    print("  MODE 1  ·  Standard — Top-5 Built-in English Voices")
    print(f"{'='*60}")

    summary: list = []

    for voice in TOP_5_VOICES:
        pt_path = os.path.join(VOICES_DIR, f"{voice}.pt")
        if not os.path.exists(pt_path):
            print(f"\n  ⚠️   Voice preset not found: {pt_path}")
            print(      "      → Ensure VibeVoice repo is cloned and voice files are present.")
            continue

        print(f"\n  🎙️  Voice : {voice}")
        try:
            prefilled   = load_voice_preset(pt_path)
            t0          = time.perf_counter()
            audio, ttfa = synthesize_long(model, processor, text, prefilled)
            elapsed     = time.perf_counter() - t0

            out_path = f"output_{voice}.wav"
            save_wav(out_path, audio)
            print_report(voice, text, audio, elapsed, ttfa)

            duration = len(audio) / SAMPLE_RATE
            rtf      = elapsed / duration if duration > 0 else float("inf")
            summary.append({"voice": voice, "path": out_path, "rtf": rtf, "dur": duration})

        except Exception as exc:
            print(f"  ❌  Synthesis failed for {voice}: {exc}")
            traceback.print_exc()

    if summary:
        print(f"\n{'='*60}")
        print("  STANDARD MODE — Final Summary")
        print(f"{'='*60}")
        print(f"  {'Voice':<22} {'Dur':>6}  {'RTF':>8}  Output file")
        print(f"  {'─'*22} {'─'*6}  {'─'*8}  {'─'*30}")
        for r in summary:
            flag = "✅" if r["rtf"] <= 1.0 else "⏳"
            print(f"  {r['voice']:<22} {r['dur']:>5.1f}s  {r['rtf']:>7.3f}x {flag}  {r['path']}")


# ──────────────────────────────────────────────────────────────────────────────
# MODE 2 — CLONING
# ──────────────────────────────────────────────────────────────────────────────
def run_cloning_mode(processor, model, text: str) -> None:
    print(f"\n{'='*60}")
    print("  MODE 2  ·  Cloning — Zero-Shot Voice from reference.wav")
    print(f"{'='*60}")

    if not os.path.exists(REF_AUDIO):
        print(f"  ℹ️   {REF_AUDIO} not found — skipping cloning mode.")
        print(      "      Place a clean, single-speaker WAV as 'reference.wav' to enable this.")
        return

    print(f"  📂  Reference : {REF_AUDIO}")
    try:
        prefilled   = build_voice_preset_from_wav(model, processor, REF_AUDIO)
        print(      "  🎙️  Synthesising with cloned voice...")
        t0          = time.perf_counter()
        audio, ttfa = synthesize_long(model, processor, text, prefilled)
        elapsed     = time.perf_counter() - t0

        save_wav("output_cloned.wav", audio)
        print_report("cloned", text, audio, elapsed, ttfa)

    except Exception as exc:
        print(f"  ❌  Cloning failed: {exc}")
        traceback.print_exc()


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 60)
    print("  VibeVoice-TTS  |  Quick-Look Evaluation Script")
    print(f"  Model  : {MODEL_PATH}")
    print(f"  Device : {DEVICE.upper()}")
    print(f"  Steps  : {DDPM_STEPS}  (CFG scale = {CFG_SCALE})")
    print("=" * 60)

    # Read & normalise input text
    if not os.path.exists(INPUT_TXT):
        sample = (
            "Welcome to VibeVoice — a long-form, multi-speaker text-to-speech "
            "system built on a next-token diffusion framework. "
            "It can synthesise natural, expressive speech for up to ninety minutes "
            "while maintaining perfect speaker consistency throughout the conversation."
        )
        with open(INPUT_TXT, "w", encoding="utf-8") as f:
            f.write(sample)
        print(f"  ℹ️   Created sample {INPUT_TXT}")

    with open(INPUT_TXT, "r", encoding="utf-8") as f:
        text = f.read().strip()

    text = (
        text.replace("\u2018", "'").replace("\u2019", "'")
            .replace("\u201c", '"').replace("\u201d", '"')
    )
    preview = text[:110] + ("…" if len(text) > 110 else "")
    print(f"\n  Input : \"{preview}\"  ({len(text)} chars / {len(text.split())} words)\n")

    # Load model
    try:
        processor, model = load_model(MODEL_PATH, DEVICE)
    except Exception as exc:
        print(f"\n[FATAL] Model load failed: {exc}")
        traceback.print_exc()
        raise SystemExit(1)

    # Run both modes
    run_standard_mode(processor, model, text)
    run_cloning_mode(processor, model, text)

    print(f"\n{'='*60}")
    print("  ✅  Evaluation complete.")
    print("  Check the *.wav files for audio quality inspection.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
