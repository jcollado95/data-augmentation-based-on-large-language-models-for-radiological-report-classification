import os
import torch
import numpy as np

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import TrainingArguments
from transformers import AutoModelForSequenceClassification
from transformers import Trainer
from sklearn.metrics import classification_report, accuracy_score
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

def multilabel_binarizer(example):
    return {"label": torch.Tensor(mlb.transform(example["chapters"]))}

set_seed()

ckpt = "checkpoints/BPNiRadBERTa_hp_search_augmented/run-1/checkpoint-9000"

tokenizer = AutoTokenizer.from_pretrained(ckpt)
model = AutoModelForSequenceClassification.from_pretrained(
    ckpt, 
    num_labels=16,
    problem_type="multi_label_classification"
)

cares = load_dataset("hf-datasets/CARES")
cares = cares.remove_columns(["icd10", "general", "area"])

cares_tokenized = cares.map(
    lambda example: tokenizer(example["full_text"], truncation=True), 
    batched=True
)

mlb = MultiLabelBinarizer()
mlb.fit(cares_tokenized["train"]["chapters"])
cares_tokenized = cares_tokenized.map(multilabel_binarizer, batched=True)

test_args = TrainingArguments(
    output_dir = "output",
    do_train = False,
    do_predict = True
)

trainer = Trainer(
    model,
    args=test_args,
    data_collator=DataCollatorWithPadding(tokenizer),
    tokenizer=tokenizer
)

test_results = trainer.predict(cares_tokenized["test"])

y_true = cares_tokenized["test"]["label"]

# --- Multilabel predictions to Multiclass predictions approach ---
probs = torch.sigmoid(torch.from_numpy(test_results.predictions)).numpy()
y_pred = (probs>0.5).astype(np.float32)

print(classification_report(y_true, y_pred, digits=4))
print("Accuracy:", accuracy_score(y_true, y_pred))



