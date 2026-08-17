import random
import re
import inspect
from pathlib import Path
from typing import Tuple

import numpy as np

from .config import DIALOGUE_LABELS, FALLBACK_LABEL


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def parse_label(raw_output: str) -> Tuple[str, bool]:
    """Parse a dialogue-act label and report whether fallback was used."""
    text = (raw_output or "").strip().lower()
    candidates = re.findall(r"[a-z]+", text)

    for token in candidates:
        if token in DIALOGUE_LABELS:
            return token, False

    for label in DIALOGUE_LABELS:
        if re.search(rf"\b{re.escape(label)}\b", text):
            return label, False

    return FALLBACK_LABEL, True


def ensure_results_dirs() -> None:
    from .config import (
        STAGE2_DIR,
        STRATEGY_A_DIR,
        STRATEGY_A_HELDOUT_DIR,
        STRATEGY_B_DIR,
        STRATEGY_B_HELDOUT_DIR,
        STRATEGY_C_DIR,
    )

    for path in (
        STRATEGY_A_DIR,
        STRATEGY_A_HELDOUT_DIR,
        STRATEGY_B_DIR,
        STRATEGY_B_HELDOUT_DIR,
        STRATEGY_C_DIR,
        STAGE2_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def resolve_model_path(model_id: str, source: str = "modelscope") -> str:
    if source == "modelscope":
        try:
            from modelscope import snapshot_download
        except ImportError as exc:
            raise ImportError(
                "modelscope is required for ModelScope model downloads. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc
        kwargs = {}
        if "ignore_file_pattern" in inspect.signature(snapshot_download).parameters:
            kwargs["ignore_file_pattern"] = [
                r"original/.*",
                r".*\.pth",
                r".*\.pth\.incomplete",
                r".*\.h5",
                r".*\.msgpack",
                r".*\.ot",
            ]
        return snapshot_download(model_id, **kwargs)

    if source == "huggingface":
        return model_id

    if source == "local":
        path = Path(model_id).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Local model path does not exist: {path}")
        return str(path)

    raise ValueError("model_source must be one of: modelscope, huggingface, local")
