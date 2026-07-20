from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"

GOLD_CSV = DATA_DIR / "cmv_300_gold_final.csv"
RAW_UTTERANCES_JSONL = DATA_DIR / "winning-args-corpus" / "utterances.jsonl"
SILVER_CANDIDATES_CSV = DATA_DIR / "task3_silver_candidates_10k.csv"
SILVER_LABELED_CSV = DATA_DIR / "task3_silver_labeled_10k.csv"

STRATEGY_A_DIR = RESULTS_DIR / "strategy_a"
STRATEGY_B_DIR = RESULTS_DIR / "strategy_b"
STRATEGY_C_DIR = RESULTS_DIR / "strategy_c"

DIALOGUE_LABELS = ["agree", "disagree", "question", "statement"]
LABEL2ID = {label: idx for idx, label in enumerate(DIALOGUE_LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}
FALLBACK_LABEL = "statement"

SEEDS = [42, 123, 2026]
SAMPLE_SEEDS = [42, 123]
SILVER_CANDIDATE_SIZE = 10000
STRATEGY_B_SIZES = [500, 1000, 1500, 2000, 2500, 3000, 5000, 8000, 10000]


@dataclass(frozen=True)
class LLMConfig:
    model_id: str = "LLM-Research/Meta-Llama-3.1-8B-Instruct"
    model_source: str = "modelscope"
    max_new_tokens: int = 8
    temperature: float = 0.0
    top_p: float = 1.0


@dataclass(frozen=True)
class TrainingConfig:
    model_name: str = "AI-ModelScope/roberta-base"
    model_source: str = "modelscope"
    learning_rate: float = 2e-5
    batch_size: int = 8
    epochs: int = 3
    weight_decay: float = 0.01
    optimizer: str = "adamw_torch"
    max_length: int = 256
    validation_size: float = 0.15
    warmup_steps: int = 0
    use_class_weights: bool = True


LLM = LLMConfig()
TRAINING = TrainingConfig()
