"""
Fix StyleTTS2 IndexError: invalid index of a 0-dim tensor at pred_dur[i].

Cause: duration.squeeze() removes all size-1 dims, so pred_dur becomes a scalar tensor
and pred_dur[i] is invalid. Some segments (e.g. short or non-Latin phonemization) hit this.

Replaces StyleTTS2.inference and long_inference_segment on the loaded styletts2.tts module.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import scipy
import torch
from cached_path import cached_path
from nltk.tokenize import word_tokenize

logger = logging.getLogger(__name__)

_PATCH_FLAG = "_ai_voice_dubber_pred_dur_fix_applied"


def _normalize_pred_dur(duration_reduced: torch.Tensor, seq_len: int) -> torch.Tensor:
    pred = torch.round(duration_reduced.reshape(-1)).clamp(min=1)
    if pred.numel() == seq_len:
        return pred
    total = max(seq_len, int(pred.sum().item()))
    out = torch.ones(seq_len, device=pred.device, dtype=pred.dtype)
    r = total - seq_len
    if r > 0:
        out = out + r // seq_len
        rem = r % seq_len
        if rem:
            out[:rem] = out[:rem] + 1
    return out


def _build_pred_aln_trg(
    input_lengths: torch.Tensor, duration_proj_out: torch.Tensor, device: torch.device
) -> torch.Tensor:
    seq_len = int(input_lengths.reshape(-1)[0].item())
    duration_reduced = torch.sigmoid(duration_proj_out).sum(axis=-1)
    pred_dur = _normalize_pred_dur(duration_reduced, seq_len)
    n_frames = int(pred_dur.sum().item())
    pred_aln_trg = torch.zeros(
        seq_len, n_frames, device=device, dtype=duration_proj_out.dtype
    )
    c_frame = 0
    for i in range(seq_len):
        d = int(pred_dur[i].item())
        pred_aln_trg[i, c_frame : c_frame + d] = 1
        c_frame += d
    return pred_aln_trg


def _patched_inference(
    self: Any,
    text: str,
    target_voice_path: Any = None,
    output_wav_file: Any = None,
    output_sample_rate: int = 24000,
    alpha: float = 0.3,
    beta: float = 0.7,
    diffusion_steps: int = 5,
    embedding_scale: float = 1,
    ref_s: Any = None,
    phonemize: bool = True,
) -> Any:
    tts_mod = __import__("styletts2.tts", fromlist=["tts"])
    SINGLE_INFERENCE_MAX_LEN = tts_mod.SINGLE_INFERENCE_MAX_LEN
    TextCleaner = tts_mod.TextCleaner
    length_to_mask = tts_mod.length_to_mask
    DEFAULT_TARGET_VOICE_URL = tts_mod.DEFAULT_TARGET_VOICE_URL

    if len(text) > SINGLE_INFERENCE_MAX_LEN:
        return self.long_inference(
            text,
            target_voice_path=target_voice_path,
            output_wav_file=output_wav_file,
            output_sample_rate=output_sample_rate,
            alpha=alpha,
            beta=beta,
            diffusion_steps=diffusion_steps,
            embedding_scale=embedding_scale,
            ref_s=ref_s,
            phonemize=phonemize,
        )

    if ref_s is None:
        if not target_voice_path or not Path(target_voice_path).exists():
            print("Cloning default target voice...")
            target_voice_path = cached_path(DEFAULT_TARGET_VOICE_URL)
        ref_s = self.compute_style(target_voice_path)

    if phonemize:
        text = text.strip()
        text = text.replace('"', "")
        phonemized_text = self.phoneme_converter.phonemize(text)
        ps = word_tokenize(phonemized_text)
        phoneme_string = " ".join(ps)
    else:
        phoneme_string = text

    textcleaner = TextCleaner()
    tokens = textcleaner(phoneme_string)
    tokens.insert(0, 0)
    tokens = torch.LongTensor(tokens).to(self.device).unsqueeze(0)

    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(self.device)
        text_mask = length_to_mask(input_lengths).to(self.device)

        t_en = self.model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = self.model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)

        s_pred = self.sampler(
            noise=torch.randn((1, 256)).unsqueeze(1).to(self.device),
            embedding=bert_dur,
            embedding_scale=embedding_scale,
            features=ref_s,
            num_steps=diffusion_steps,
        ).squeeze(1)

        s = s_pred[:, 128:]
        ref = s_pred[:, :128]

        ref = alpha * ref + (1 - alpha) * ref_s[:, :128]
        s = beta * s + (1 - beta) * ref_s[:, 128:]

        d = self.model.predictor.text_encoder(d_en, s, input_lengths, text_mask)

        x, _ = self.model.predictor.lstm(d)
        duration = self.model.predictor.duration_proj(x)
        pred_aln_trg = _build_pred_aln_trg(input_lengths, duration, self.device)

        en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(self.device))
        if self.model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(en)
            asr_new[:, :, 0] = en[:, :, 0]
            asr_new[:, :, 1:] = en[:, :, 0:-1]
            en = asr_new

        F0_pred, N_pred = self.model.predictor.F0Ntrain(en, s)

        asr = t_en @ pred_aln_trg.unsqueeze(0).to(self.device)
        if self.model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(asr)
            asr_new[:, :, 0] = asr[:, :, 0]
            asr_new[:, :, 1:] = asr[:, :, 0:-1]
            asr = asr_new

        out = self.model.decoder(asr, F0_pred, N_pred, ref.squeeze().unsqueeze(0))

    output = out.squeeze().cpu().numpy()[..., :-50]
    if output_wav_file:
        scipy.io.wavfile.write(output_wav_file, rate=output_sample_rate, data=output)
    return output


def _patched_long_inference_segment(
    self: Any,
    text: Any,
    prev_s: Any,
    ref_s: Any,
    alpha: float = 0.3,
    beta: float = 0.7,
    t: float = 0.7,
    diffusion_steps: int = 5,
    embedding_scale: float = 1,
    phonemize: bool = True,
) -> Tuple[Any, Any]:
    tts_mod = __import__("styletts2.tts", fromlist=["tts"])
    TextCleaner = tts_mod.TextCleaner
    length_to_mask = tts_mod.length_to_mask

    if phonemize:
        text = text.strip()
        text = text.replace('"', "")
        phonemized_text = self.phoneme_converter.phonemize(text)
        ps = word_tokenize(phonemized_text)
        phoneme_string = " ".join(ps)
        phoneme_string = phoneme_string.replace("``", '"')
        phoneme_string = phoneme_string.replace("''", '"')
    else:
        phoneme_string = text

    textcleaner = TextCleaner()
    tokens = textcleaner(phoneme_string)
    tokens.insert(0, 0)
    tokens = torch.LongTensor(tokens).to(self.device).unsqueeze(0)

    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(self.device)
        text_mask = length_to_mask(input_lengths).to(self.device)

        t_en = self.model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = self.model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)

        s_pred = self.sampler(
            noise=torch.randn((1, 256)).unsqueeze(1).to(self.device),
            embedding=bert_dur,
            embedding_scale=embedding_scale,
            features=ref_s,
            num_steps=diffusion_steps,
        ).squeeze(1)

        if prev_s is not None:
            s_pred = t * prev_s + (1 - t) * s_pred

        s = s_pred[:, 128:]
        ref = s_pred[:, :128]

        ref = alpha * ref + (1 - alpha) * ref_s[:, :128]
        s = beta * s + (1 - beta) * ref_s[:, 128:]

        s_pred = torch.cat([ref, s], dim=-1)

        d = self.model.predictor.text_encoder(d_en, s, input_lengths, text_mask)

        x, _ = self.model.predictor.lstm(d)
        duration = self.model.predictor.duration_proj(x)
        pred_aln_trg = _build_pred_aln_trg(input_lengths, duration, self.device)

        en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(self.device))
        if self.model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(en)
            asr_new[:, :, 0] = en[:, :, 0]
            asr_new[:, :, 1:] = en[:, :, 0:-1]
            en = asr_new

        F0_pred, N_pred = self.model.predictor.F0Ntrain(en, s)

        asr = t_en @ pred_aln_trg.unsqueeze(0).to(self.device)
        if self.model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(asr)
            asr_new[:, :, 0] = asr[:, :, 0]
            asr_new[:, :, 1:] = asr[:, :, 0:-1]
            asr = asr_new

        out = self.model.decoder(asr, F0_pred, N_pred, ref.squeeze().unsqueeze(0))

    return out.squeeze().cpu().numpy()[..., :-100], s_pred


def apply_styletts_pred_dur_fix(tts_mod: Any) -> None:
    """Monkey-patch styletts2.tts.StyleTTS2 before any instance is created."""
    cls = tts_mod.StyleTTS2
    if getattr(cls, _PATCH_FLAG, False):
        return
    cls.inference = _patched_inference  # type: ignore[assignment]
    cls.long_inference_segment = _patched_long_inference_segment  # type: ignore[assignment]
    setattr(cls, _PATCH_FLAG, True)
    logger.info("[styletts] Patched StyleTTS2 inference (pred_dur / pred_aln_trg).")
