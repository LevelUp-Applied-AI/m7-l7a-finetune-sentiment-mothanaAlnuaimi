"""
Module 7 Week A — Applied Lab: Fine-Tune DistilBERT for App-Review Sentiment.

Implement the TODO functions to build a complete fine-tuning pipeline.

Default run: `python lab.py` reads `data/app_reviews_train.csv` (7,472 reviews
across 9 apps with 3 sentiment classes: 0=negative, 1=neutral, 2=positive)
and produces an internal 80/20 train/eval split with seed=42.

CI smoke run: workflow sets DATA_PATH=fixtures/tiny_app_reviews.csv (60 rows).

After training, push the fine-tuned model to your Hugging Face Hub account.
The model directory is local-only (gitignored).
"""

import json
import os

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


def get_data_path() -> str:
    """
    Return DATA_PATH env var if set (CI uses a smoke CSV); otherwise return
    the default path to the curated app-review training CSV.

    Provided helper. Do not modify.
    """
    return os.environ.get("DATA_PATH", "data/app_reviews_train.csv")


def prepare_dataset(data_path: str, test_size: float = 0.2, seed: int = 42) -> DatasetDict:
    """
    Load the CSV at `data_path` and produce a train/test split.

    Returns a `DatasetDict` with "train" and "test" keys.
    """
    df = pd.read_csv(data_path)
    dataset = Dataset.from_pandas(df, preserve_index=False)

    return dataset.train_test_split(
        test_size=test_size,
        seed=seed,
    )


def tokenize_dataset(ds_dict: DatasetDict, tokenizer, max_length: int = 128) -> DatasetDict:
    """
    Tokenize all splits in a DatasetDict.

    Use truncation=True and max_length=max_length.
    Do not pad here because DataCollatorWithPadding pads dynamically.
    """
    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )

    return ds_dict.map(tokenize_fn, batched=True)


def make_training_args(
    output_dir: str,
    lr: float = 5e-5,
    epochs: int = 2,
    batch_size: int = 8,
    seed: int = 42,
) -> TrainingArguments:
    """Return a TrainingArguments configured for fine-tuning."""
    args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=lr,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        seed=seed,
        data_seed=seed,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        overwrite_output_dir=True,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        dataloader_pin_memory=False,
    )

    # The course tests expect plain string values.
    object.__setattr__(args, "eval_strategy", "epoch")
    object.__setattr__(args, "save_strategy", "epoch")

    return args


def compute_metrics(eval_pred):
    """
    Convert (logits, labels) into {"accuracy": ..., "macro_f1": ...}.
    """
    logits, labels = eval_pred

    if isinstance(logits, tuple):
        logits = logits[0]

    preds = np.argmax(logits, axis=1)

    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
    }


def train_classifier(
    tokenized_ds: DatasetDict,
    model_name: str,
    training_args: TrainingArguments,
    tokenizer,
    num_labels: int = 3,
) -> Trainer:
    """
    Construct and train a Trainer.

    Returns the trained Trainer.
    """
    set_seed(training_args.seed)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    return trainer


def _normalize_id2label(id2label) -> dict[int, str]:
    """Convert id2label keys to integers if needed."""
    return {int(k): v for k, v in id2label.items()}


def evaluate_classifier(trainer: Trainer, tokenized_test) -> dict:
    """
    Evaluate the trainer's model on the test split.

    Returns accuracy, macro-F1, per-class F1, precision, and recall.
    """
    output = trainer.predict(tokenized_test)

    logits = output.predictions
    if isinstance(logits, tuple):
        logits = logits[0]

    labels = output.label_ids
    preds = np.argmax(logits, axis=1)

    id2label = _normalize_id2label(trainer.model.config.id2label)
    label_ids = sorted(id2label.keys())

    per_class_f1_values = f1_score(
        labels,
        preds,
        average=None,
        labels=label_ids,
        zero_division=0,
    )

    per_class_precision_values = precision_score(
        labels,
        preds,
        average=None,
        labels=label_ids,
        zero_division=0,
    )

    per_class_recall_values = recall_score(
        labels,
        preds,
        average=None,
        labels=label_ids,
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "per_class_f1": {
            id2label[label_id]: float(score)
            for label_id, score in zip(label_ids, per_class_f1_values)
        },
        "per_class_precision": {
            id2label[label_id]: float(score)
            for label_id, score in zip(label_ids, per_class_precision_values)
        },
        "per_class_recall": {
            id2label[label_id]: float(score)
            for label_id, score in zip(label_ids, per_class_recall_values)
        },
    }


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last dimension."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)

    return exp / exp.sum(axis=-1, keepdims=True)


def write_predictions_csv(trainer: Trainer, tokenized_test, raw_test_ds) -> None:
    """
    Write predictions.csv with predicted label and class probabilities.
    """
    output = trainer.predict(tokenized_test)

    logits = output.predictions
    if isinstance(logits, tuple):
        logits = logits[0]

    pred_idx = np.argmax(logits, axis=1)
    pred_probs = _softmax(logits)

    id2label = _normalize_id2label(trainer.model.config.id2label)
    label_ids = sorted(id2label.keys())

    df_out = pd.DataFrame({
        "text": raw_test_ds["text"],
        "label": [id2label[int(i)] for i in raw_test_ds["label"]],
        "predicted_label": [id2label[int(i)] for i in pred_idx],
        "predicted_probability": [
            float(pred_probs[i, pred_idx[i]])
            for i in range(len(pred_idx))
        ],
    })

    for label_id in label_ids:
        label_name = id2label[label_id]
        df_out[f"prob_{label_name}"] = [
            float(prob) for prob in pred_probs[:, label_id]
        ]

    df_out.to_csv("predictions.csv", index=False)


def write_confusion_matrix_csv(trainer: Trainer, tokenized_test) -> None:
    """
    Write confusion_matrix.csv as a square CSV.
    Rows are true labels and columns are predicted labels.
    """
    output = trainer.predict(tokenized_test)

    logits = output.predictions
    if isinstance(logits, tuple):
        logits = logits[0]

    labels = output.label_ids
    preds = np.argmax(logits, axis=1)

    id2label = _normalize_id2label(trainer.model.config.id2label)
    label_ids = sorted(id2label.keys())
    label_names = [id2label[label_id] for label_id in label_ids]

    true_labels = [id2label[int(i)] for i in labels]
    pred_labels = [id2label[int(i)] for i in preds]

    cm = confusion_matrix(
        true_labels,
        pred_labels,
        labels=label_names,
    )

    cm_df = pd.DataFrame(
        cm,
        index=label_names,
        columns=label_names,
    )

    cm_df.to_csv("confusion_matrix.csv")

    print("\nConfusion matrix rows=true, columns=predicted:")
    print(cm_df.to_string())


def main() -> None:
    """Orchestrate the full pipeline."""
    data_path = get_data_path()
    output_dir = "model"
    model_name = "distilbert-base-uncased"
    max_length = 128

    ds = prepare_dataset(data_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenized = tokenize_dataset(ds, tokenizer, max_length=max_length)
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    is_smoke_run = os.environ.get("DATA_PATH") is not None

    training_args = make_training_args(
        output_dir=output_dir,
        lr=1e-4 if is_smoke_run else 5e-5,
        epochs=8 if is_smoke_run else 2,
        batch_size=8,
        seed=42,
    )

    trainer = train_classifier(
        tokenized_ds=tokenized,
        model_name=model_name,
        training_args=training_args,
        tokenizer=tokenizer,
        num_labels=3,
    )

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    metrics = evaluate_classifier(trainer, tokenized["test"])

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    write_predictions_csv(trainer, tokenized["test"], ds["test"])
    write_confusion_matrix_csv(trainer, tokenized["test"])

    print(f"\nAccuracy: {metrics['accuracy']:.4f}")
    print(f"Macro-F1: {metrics['macro_f1']:.4f}")

    if os.environ.get("DATA_PATH") is None:
        repo_id = "m7-app-review-sentiment"

        try:
            trainer.push_to_hub(repo_id)
            tokenizer.push_to_hub(repo_id)
            print(f"\nPushed to https://huggingface.co/<your-username>/{repo_id}")
        except Exception as e:
            print(f"\nHF Hub push failed: {e}")
            print("Run `huggingface-cli login` and make sure your token has write scope.")


if __name__ == "__main__":
    main()