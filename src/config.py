from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"

GOLD_CSV = DATA_DIR / "cmv_300_gold_final.csv"
RAW_UTTERANCES_JSONL = DATA_DIR / "winning-args-corpus" / "utterances.jsonl"
SILVER_CANDIDATES_CSV = DATA_DIR / "task3_silver_candidates_10k.csv"
SILVER_LABELED_CSV = DATA_DIR / "task3_silver_labeled_10k.csv"

# Held-out test set: independent of GOLD_CSV, never used for tuning/model
# selection. GOLD_CSV plays the dev role (training + hyperparameter/config
# selection for all three strategies); TEST_CSV is only touched once per
# strategy, after its config below has been locked in.
# Expanded from the original 80-row pilot (pilot80 + batch2, double-annotated
# with Cohen's kappa 0.9563) for the A-vs-B/A-vs-C power analysis. Batch2 was
# collected via simple random sampling (natural class distribution), so the
# 530-row union is class-imbalanced (disagree 188 / statement 164 / question
# 95 / agree 83); src/prepare_heldout_balanced.py downsamples every class to
# the smallest class's count (83) for a balanced 332-row set. Per-class macro-F1
# precision is bottlenecked by the smallest class regardless, so this loses no
# effective power. The full imbalanced set is still available as
# cmv_test_candidates_heldout_expanded.csv. The pre-expansion *_heldout
# results (n=80) remain in results/strategy_*_heldout for comparison.
TEST_CSV = DATA_DIR / "cmv_test_candidates_heldout_balanced.csv"

STRATEGY_A_DIR = RESULTS_DIR / "strategy_a"
STRATEGY_B_DIR = RESULTS_DIR / "strategy_b_zeroshot"
STRATEGY_C_DIR = RESULTS_DIR / "strategy_c"
STRATEGY_A_HELDOUT_DIR = RESULTS_DIR / "strategy_a_heldout"
STRATEGY_B_HELDOUT_DIR = RESULTS_DIR / "strategy_b_heldout"
STRATEGY_C_HELDOUT_DIR = RESULTS_DIR / "strategy_c_heldout"

# Locked final configs, chosen from GOLD_CSV (dev) results. Each was picked
# by macro_f1_mean in results/<strategy>/summary*.csv before TEST_CSV was
# ever evaluated against. Do not add list-valued fields here — heldout eval
# functions must only accept one fixed config, never a sweep.
STRATEGY_A_FINAL_MODE = "fewshot"  # zeroshot=0.5148 F1, fewshot=0.5916 F1 on gold

STRATEGY_B_FINAL_SILVER_CSV = DATA_DIR / "task3_silver_labeled_10k_fewshot.csv"
STRATEGY_B_FINAL_CONFIG = {
    "silver_size": 2500,
    "sample_seed": 123,
    "use_class_weights": True,
}  # sample_seed=123 macro_f1_mean=0.6375 (vs sample_seed=42: 0.6064) on gold;
# silver_size=2500 was the top overall (0.6220 avg across both sample_seeds)
# in the fewshot-silver sweep in results/strategy_b_fewshot/summary.csv

STRATEGY_C_FINAL_CONFIG = {
    "model_type": "roberta",
    "use_class_weights": True,
    "split_seed": 42,
    "warmup_steps": 20,
    "epochs": 8,
}  # gold 5-fold OOF macro_f1_mean=0.5309 (std 0.0038), zero class collapse across
# all 15 fold/seed runs, in results/strategy_c/roberta_weights_warmup20_epochs8/
# summary_oof.csv. Beats TF-IDF+weights (0.2485) and weights-off at the same
# warmup/epochs (0.3499, collapses on 8/15 runs). The warmup=20/epochs=8 fix
# resolved the training-instability diagnosed in strategy_c_diagnose (models
# were failing to fit their own training folds); earlier collapsed numbers
# from before that fix (RoBERTa+weights 0.2208, no-weights 0.1629) no longer
# apply and should not be cited.

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
