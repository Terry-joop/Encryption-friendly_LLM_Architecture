"""Compute downstream task metrics from HE eval output."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

DEFAULT_LABELS = {
    "mrpc": REPO_ROOT / "plaintext/fine-tuning_data/labels_mrpc_eval.pth",
    "rte": REPO_ROOT / "plaintext/fine-tuning_data/labels_rte_eval.pth",
}


def numeric_rows(file_path):
    rows = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            cleaned = line.strip().rstrip(",")
            if not cleaned:
                continue
            try:
                values = [float(value.strip()) for value in cleaned.split(",")]
            except ValueError:
                continue
            if len(values) >= 3:
                rows.append(values[:3])
    return rows


def sort_by_blocks(rows, block_size):
    if block_size <= 1:
        return rows

    sorted_rows = []
    for idx in range(0, len(rows), block_size):
        block = rows[idx : idx + block_size]
        sorted_rows.extend(sorted(block, key=lambda row: row[0]))
    return sorted_rows


def predictions_from_rows(rows, invert=False):
    preds = [0 if row[1] > row[2] else 1 for row in rows]
    if invert:
        preds = [1 - pred for pred in preds]
    return preds


def load_labels(path, count):
    labels = torch.load(path).numpy().tolist()
    return labels[:count]


def print_classification_metrics(task, labels, preds, title):
    print(f"[{title}]")
    print(f"num_eval_outputs = {len(preds)}")
    print(f"true label counts = {dict(Counter(labels))}")
    print(f"pred counts = {dict(Counter(preds))}")
    print(f"accuracy = {accuracy_score(labels, preds):.10f}")

    if task == "mrpc":
        print(f"f1 = {f1_score(labels, preds):.10f}")
    elif task == "cola":
        print(f"mcc = {matthews_corrcoef(labels, preds):.10f}")
    else:
        if len(set(labels)) <= 2 and len(set(preds)) <= 2:
            print(f"f1 = {f1_score(labels, preds):.10f}")

    print(f"confusion_matrix = {confusion_matrix(labels, preds).tolist()}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="mrpc", help="Task name: mrpc, rte, cola, sst2, qnli")
    parser.add_argument("--pred-file", required=True, help="HE eval output log/txt file")
    parser.add_argument("--label-file", help="Evaluation label .pth file")
    parser.add_argument(
        "--block-size",
        type=int,
        default=1,
        help="Rows per eval output block. Use 1 for one GPU, 8 for 8 GPUs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    task = args.task.lower()
    pred_file = Path(args.pred_file)
    label_file = args.label_file or DEFAULT_LABELS.get(task)
    if label_file is None:
        raise ValueError("--label-file is required for this task")

    rows = sort_by_blocks(numeric_rows(pred_file), args.block_size)
    normal_preds = predictions_from_rows(rows, invert=False)
    labels = load_labels(label_file, len(normal_preds))

    print(f"task = {task}")
    print(f"pred_file = {pred_file}")
    print(f"label_file = {label_file}")
    print(f"block_size = {args.block_size}")
    print()

    print_classification_metrics(task, labels, normal_preds, "normal")
    print()
    inverted_preds = predictions_from_rows(rows, invert=True)
    print_classification_metrics(task, labels, inverted_preds, "inverted_labels")


if __name__ == "__main__":
    main()
