import numpy as np
import os
import torch
import ast

from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import TrainingArguments
from transformers import AutoModelForSequenceClassification
from transformers import Trainer
from transformers import EarlyStoppingCallback
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer


def set_seed(seed=777):
    """Seed for reproducible experiments."""
    os.environ["SEED"] = str(seed)
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    np.random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def compute_metrics(eval_pred):
    """Computes metrics during evaluation.
    
    Returns:
        A dictionary with the name of the metrics as keys and their score as float values."""

    logits, labels = eval_pred
    probs = torch.sigmoid(torch.from_numpy(logits)).numpy()
    predictions = (probs>0.5).astype(np.float32)

    micro_precision = precision_score(y_true=labels, y_pred=predictions, average='micro')
    macro_precision = precision_score(y_true=labels, y_pred=predictions, average='macro')
    micro_recall = recall_score(y_true=labels, y_pred=predictions, average='micro')
    macro_recall = recall_score(y_true=labels, y_pred=predictions, average='macro')
    micro_f1 = f1_score(y_true=labels, y_pred=predictions, average='micro')
    macro_f1 = f1_score(y_true=labels, y_pred=predictions, average='macro')

    return {
        "micro_precision": micro_precision,
        "macro_precision": macro_precision,
        "micro_recall": micro_recall,
        "macro_recall": macro_recall,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1
    }

def multilabel_binarizer(example):
    return {"label": torch.Tensor(mlb.transform(example["chapters"]))}

def wandb_setup():
    os.environ["WANDB_PROJECT"] = "TFM"
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_LOG_MODEL"] = "true"
    os.environ["WANDB_WATCH"] = "all"

def optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 3e-5, 5e-5, log=True),
        "seed": trial.suggest_int("seed", 320, 327),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32]),
        "per_device_eval_batch_size": trial.suggest_categorical("per_device_eval_batch_size", [8, 16, 32]),
        "weight_decay": trial.suggest_float("weight_decay", 1e-12, 1e-1, log=True),
        "adam_epsilon": trial.suggest_float("adam_epsilon", 1e-10, 1e-6, log=True),
        "warmup_steps": trial.suggest_int('warmup_steps', 0, 1000)
    }

def compute_objective(metrics):
    return metrics["eval_macro_f1"]

def model_init():
    return AutoModelForSequenceClassification.from_pretrained(
        checkpoint, 
        num_labels=16,
        problem_type="multi_label_classification"
    )

set_seed()
wandb_setup()

checkpoint = "hf-models/roberta-base-biomedical-clinical-es"
tokenizer_ckpt = "hf-models/roberta-base-biomedical-clinical-es"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_ckpt, problem_type="multi_label_classification")

cares = load_dataset("hf-datasets/CARES")
cares = cares.remove_columns(["icd10", "general", "area"])

cares_aug = load_dataset("csv", data_files="data/cares_aug.tsv", sep="\t")
cares_aug = cares_aug.map(lambda x: {"chapters": ast.literal_eval(x["chapters"])})
cares["train"] = concatenate_datasets([cares["train"], cares_aug["train"]])

cares_tokenized = cares.map(
    lambda example: tokenizer(example["full_text"], truncation=True), 
    batched=True
)

mlb = MultiLabelBinarizer()
mlb.fit(cares_tokenized["train"]["chapters"])
cares_tokenized = cares_tokenized.map(multilabel_binarizer, batched=True)

training_args = TrainingArguments(
    output_dir=f"checkpoints/RoBERTa_hp_search_augmented/",
    evaluation_strategy="steps",
    num_train_epochs=100,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=1,
    metric_for_best_model='eval_macro_f1',
    load_best_model_at_end=True,
    logging_first_step=True,
    logging_steps=100,
    logging_strategy="steps",
    report_to=["wandb"],
    run_name="RoBERTa_100_epochs_optuna_augmented"
)

trainer = Trainer(
    args=training_args,
    train_dataset=cares_tokenized["train"],
    eval_dataset=cares_tokenized["test"],
    data_collator=DataCollatorWithPadding(tokenizer),
    tokenizer=tokenizer,
    model_init=model_init,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(10)]
)

best_trial = trainer.hyperparameter_search(
    direction="maximize",
    backend="optuna",
    hp_space=optuna_hp_space,
    n_trials=10,
    compute_objective=compute_objective
)