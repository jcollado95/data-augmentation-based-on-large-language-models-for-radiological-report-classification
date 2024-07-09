import numpy as np
import os
import torch

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import TrainingArguments
from transformers import AutoModelForSequenceClassification
from transformers import Trainer
from transformers import EarlyStoppingCallback
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
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

set_seed()
wandb_setup()

checkpoint = "BPNiRadBERTa/checkpoint-12000"
tokenizer_ckpt = "BPNiRadBERTa/checkpoint-12000"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_ckpt, problem_type="multi_label_classification")

cares = load_dataset("hf-datasets/CARES")

cares_tokenized = cares.map(
    lambda example: tokenizer(example["full_text"], truncation=True), 
    batched=True
)

mlb = MultiLabelBinarizer()
mlb.fit(cares_tokenized["train"]["chapters"])
cares_tokenized = cares_tokenized.map(multilabel_binarizer, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir=f"checkpoints/BPNiRadBERTa_100_epochs/",
    evaluation_strategy="steps",
    num_train_epochs=100,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=1,
    metric_for_best_model='eval_macro_f1',
    auto_find_batch_size=True,
    load_best_model_at_end=True,
    logging_dir="logs/BPNiRadBERTa_100_epochs/",
    logging_first_step=True,
    logging_steps=100,
    logging_strategy="steps",
    report_to=["wandb", "tensorboard"],
    run_name="BPNiRadBERTa_100_epochs"
)

model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint, 
    num_labels=16,
    problem_type="multi_label_classification"
)

trainer = Trainer(
    model,
    training_args,
    train_dataset=cares_tokenized["train"],
    eval_dataset=cares_tokenized["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(10)]
)

trainer.train()

# PREDICTIONS
logits, labels, metrics = trainer.predict(cares_tokenized["test"])
print("Metrics: ", metrics)

probs = torch.sigmoid(torch.from_numpy(logits)).numpy()
predictions = (probs>0.5).astype(np.float32)

print(classification_report(
        labels, 
        predictions,
        digits=4
    )
)