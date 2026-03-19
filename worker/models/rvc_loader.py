"""
RVC: delegates to ai_models.rvc_model. Mandatory when RVC_MODEL_PATH is set; no passthrough.
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

from worker.ai_models import rvc_model


def run_rvc(
    input_wav_path: str,
    output_wav_path: str,
    speaker_wav_path: Optional[str] = None,
) -> bool:
    """Run RVC conversion. Raises if RVC not configured (RVC_MODEL_PATH). Returns True after conversion."""
    rvc_model.load_rvc()
    return rvc_model.convert_voice(input_wav_path, output_wav_path, speaker_wav_path=speaker_wav_path)
