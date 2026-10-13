from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"

GOLD_CSV = DATA_DIR / "cmv_300_gold_final.csv"
RAW_UTTERANCES_JSONL = DATA_DIR / "winning-args-corpus" / "utterances.jsonl"
SILVER_CANDIDATES_CSV = DATA_DIR / "task3_silver_candidates_10k.csv"
SILVER_LABELED_CSV = DATA_DIR / "task3_silver_labeled_10k.csv"

# GOLD_CSV is used for development and configuration selection. TEST_CSV is
# the independent held-out set and must only be used after a configuration is locked.
TEST_CSV = DATA_DIR / "cmv_test_candidates_heldout_balanced.csv"

# Fixed Intent-stratified split: dev is for prompt development; eval is for the
# final locked prompts.
STAGE2_DEV_CSV = DATA_DIR / "cmv_300_gold_stage2_dev.csv"
STAGE2_EVAL_CSV = DATA_DIR / "cmv_300_gold_stage2_eval.csv"
STAGE2_DEV_FRACTION = 0.12
STAGE2_SPLIT_SEED = 42

STRATEGY_A_DIR = RESULTS_DIR / "strategy_a"
STRATEGY_B_DIR = RESULTS_DIR / "strategy_b_zeroshot"
STRATEGY_C_DIR = RESULTS_DIR / "strategy_c"
STRATEGY_A_HELDOUT_DIR = RESULTS_DIR / "strategy_a_heldout"
STRATEGY_B_HELDOUT_DIR = RESULTS_DIR / "strategy_b_heldout"
STRATEGY_C_HELDOUT_DIR = RESULTS_DIR / "strategy_c_heldout"

# Final configurations were selected on development data before held-out testing.
# Keep each setting scalar: held-out evaluation accepts one configuration, not a sweep.
STRATEGY_A_FINAL_MODE = "fewshot"

STRATEGY_B_FINAL_SILVER_CSV = DATA_DIR / "task3_silver_labeled_10k_fewshot.csv"
STRATEGY_B_FINAL_CONFIG = {
    "silver_size": 2500,
    "sample_seed": 123,
    "use_class_weights": True,
}

STRATEGY_C_FINAL_CONFIG = {
    "model_type": "roberta",
    "use_class_weights": True,
    "split_seed": 42,
    "warmup_steps": 20,
    "epochs": 8,
}

DIALOGUE_LABELS = ["agree", "disagree", "question", "statement"]
LABEL2ID = {label: idx for idx, label in enumerate(DIALOGUE_LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}
FALLBACK_LABEL = "statement"

# Preserve the casing used by the Stage 2 gold columns.
STAGE2_DIR = RESULTS_DIR / "stage2"

SENTIMENT_LABELS = ["Positive", "Negative", "Neutral"]
FALLBACK_SENTIMENT = "Neutral"

EMOTION_LABELS = ["Sarcasm", "Hostility", "Contempt", "Neutral", "Curiosity", "Appreciation"]

INTENT_LABELS = ["Information seeking", "Challenge", "Counter-argue", "Support", "Others"]
FALLBACK_INTENT = "Others"

# Stage 2 needs more output tokens than single-label Strategy A inference.
STAGE2_MAX_NEW_TOKENS = 64

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
