"""
RVC voice conversion. GPU; use with GPU_LOCK. Converts TTS output to cloned voice.
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def convert_voice(
    input_wav_path: str,
    output_wav_path: str,
    speaker_wav_path: Optional[str] = None,
) -> bool:
    """
    If RVC is available, convert input to sound like speaker. Else copy input to output.
    Returns True if conversion was performed, False if passthrough.
    """
    from worker.models.rvc_loader import run_rvc
    return run_rvc(input_wav_path, output_wav_path, speaker_wav_path=speaker_wav_path)
