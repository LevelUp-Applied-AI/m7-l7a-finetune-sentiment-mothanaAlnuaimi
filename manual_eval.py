"""
Stretch Tuesday — Manual Evaluation Harness.

Implement these without using Trainer.predict, sklearn metrics helpers, or
Hugging Face evaluate. The goal is to make the math explicit.
"""

import numpy as np
import torch


def manual_predict(model, tokenizer, texts: list[str], batch_size: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """
    Manual PyTorch inference. No Trainer.predict.
    Returns (preds, probs):
      preds: shape (N,), int class indices
      probs: shape (N, num_classes), probabilities (post-softmax)
    """
    model.eval()

    all_probs = []
    all_preds = []

    device = next(model.parameters()).device

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]

            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )

            encoded = {key: value.to(device) for key, value in encoded.items()}

            outputs = model(**encoded)
            logits = outputs.logits

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    probs = np.concatenate(all_probs, axis=0)
    preds = np.concatenate(all_preds, axis=0)

    return preds, probs


def compute_classification_report_from_arrays(y_true, y_pred) -> dict:
    """
    Compute accuracy, per-class precision/recall/F1, and macro-F1 using
    only numpy. No sklearn metric helpers, no Hugging Face evaluate.

    Returns: {
      "accuracy": float,
      "macro_f1": float,
      "per_class": {label_index: {"precision": ..., "recall": ..., "f1": ...}, ...},
    }
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    labels = np.unique(np.concatenate([y_true, y_pred]))

    accuracy = float(np.mean(y_true == y_pred))

    per_class = {}
    f1_scores = []

    for label in labels:
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[int(label)] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

        f1_scores.append(f1)

    macro_f1 = float(np.mean(f1_scores))

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": per_class,
    }
