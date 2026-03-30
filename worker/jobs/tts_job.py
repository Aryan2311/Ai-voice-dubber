"""
Text-to-speech job: text + optional voice_sample -> XTTS -> upload speech.wav
"""
import logging
import os
import tempfile

from worker.utils import s3_utils
from worker.ai_models import xtts_model

logger = logging.getLogger(__name__)


def run_tts_job(job: dict) -> None:
    """
    job: { job_type: "TEXT_TO_SPEECH", request_id: str, text: str, language: str, voice_sample?: str }
    Output: tts/{request_id}/speech.wav
    """
    request_id = job["request_id"]
    text = job["text"]
    language = job.get("language", "en")
    voice_sample_s3 = job.get("voice_sample")
    logger.info("TEXT_TO_SPEECH request_id=%s language=%s has_voice_sample=%s", request_id, language, bool(voice_sample_s3))

    with tempfile.TemporaryDirectory() as tmp:
        speaker_wav = None
        if voice_sample_s3:
            speaker_wav = s3_utils.download_to_temp(voice_sample_s3, suffix=".wav")

        out_wav = os.path.join(tmp, "speech.wav")
        xtts_model.generate_speech(text, out_wav, language=language, speaker_wav_path=speaker_wav)

        s3_key = f"tts/{request_id}/speech.wav"
        s3_utils.upload_file(out_wav, s3_key, content_type="audio/wav")
        logger.info("TEXT_TO_SPEECH request_id=%s uploaded key=%s", request_id, s3_key)
        return s3_key
