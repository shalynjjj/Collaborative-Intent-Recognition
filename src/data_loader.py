from pathlib import Path
from typing import Optional

import pandas as pd

from .config import DIALOGUE_LABELS, GOLD_CSV


def load_gold_data(path: Optional[Path] = None) -> pd.DataFrame:
    csv_path = Path(path) if path is not None else GOLD_CSV
    df = pd.read_csv(csv_path, encoding="latin1")

    if "Int" in df.columns and "Intent" not in df.columns:
        df = df.rename(columns={"Int": "Intent"})

    required = {"Parent", "Reply", "Dialogue_act"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")

    df["Dialogue_act"] = df["Dialogue_act"].astype(str).str.strip().str.lower()
    unknown = sorted(set(df["Dialogue_act"]) - set(DIALOGUE_LABELS))
    if unknown:
        raise ValueError(
            "Unexpected Dialogue_act labels: "
            f"{unknown}. Expected exactly {DIALOGUE_LABELS}."
        )

    df["model_input"] = build_model_input(df)
    return df


def build_model_input(df: pd.DataFrame) -> pd.Series:
    parent = df["Parent"].fillna("").astype(str)
    reply = df["Reply"].fillna("").astype(str)
    return parent + " [SEP] " + reply

