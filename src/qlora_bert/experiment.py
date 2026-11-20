"""QLoRA vs. full fine-tuning of BERT on IMDB reviews, measured on the same GPU and split.

    python -m qlora_bert.experiment --method qlora --out results
    python -m qlora_bert.experiment --method full  --out results

Run each method in its own process so peak GPU memory is measured from a clean start.
Each run writes ``results/<method>.json`` with accuracy, F1, AUC, peak memory, training time,
trainable parameters and latency.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)

from qlora_bert.core import Config, classification_metrics

DATASET_ID = "dipanjanS/imdb_sentiment_finetune_dataset20k"  # 8k train / 2k validation / 10k test
MODEL_ID = "bert-base-uncased"


def build_model(config: Config) -> torch.nn.Module:
    if config.method == "full":
        return AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=2)

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=2,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        ),
        device_map={"": 0},
    )
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["query", "value"],
        task_type="SEQ_CLS",  # also keeps the new classification head trainable
    )
    return get_peft_model(model, lora)


def trainable_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.inference_mode()
def latency_ms(model: torch.nn.Module, tokenizer, text: str, repeats: int = 50) -> float:
    """Median single-review latency (batch size 1) after warm-up."""
    model.eval()
    inputs = tokenizer(text, truncation=True, max_length=512, return_tensors="pt").to("cuda")
    for _ in range(5):
        model(**inputs)
    timings = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        model(**inputs)
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000)
    return round(float(np.median(timings)), 1)


def run(config: Config, out: Path) -> dict:
    torch.manual_seed(config.seed)
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    data = load_dataset(DATASET_ID).map(
        lambda batch: tokenizer(batch["review"], truncation=True, max_length=config.max_length),
        batched=True,
        remove_columns=["review"],
    )
    data = data.rename_column("sentiment", "labels")

    model = build_model(config)
    trainable = trainable_parameters(model)
    args = TrainingArguments(
        output_dir=str(out / f"checkpoints_{config.method}"),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=64,
        learning_rate=config.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.06,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,  # best epoch by validation accuracy, never by test
        metric_for_best_model="accuracy",
        fp16=True,
        gradient_checkpointing=config.method == "qlora",
        logging_steps=50,
        report_to="none",
        seed=config.seed,
    )

    def compute_metrics(prediction: EvalPrediction) -> dict[str, float]:
        return classification_metrics(prediction.predictions, prediction.label_ids)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    started = time.perf_counter()
    trainer.train()
    train_seconds = time.perf_counter() - started
    peak_memory = torch.cuda.max_memory_allocated() / 1e9

    validation = classification_metrics(*_logits_and_labels(trainer, data["validation"]))
    test_logits, test_labels = _logits_and_labels(trainer, data["test"])
    result = {
        "config": asdict(config),
        "validation": validation,
        "test": classification_metrics(test_logits, test_labels),
        "test_errors": {
            "false_positives": int(((test_logits.argmax(1) == 1) & (test_labels == 0)).sum()),
            "false_negatives": int(((test_logits.argmax(1) == 0) & (test_labels == 1)).sum()),
        },
        "trainable_parameters": trainable,
        "peak_gpu_memory_gb": round(peak_memory, 2),
        "train_seconds": round(train_seconds, 1),
        "latency_ms_batch1": latency_ms(
            model, tokenizer, "A slow start, but the ending won me over."
        ),
        "gpu": torch.cuda.get_device_name(0),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{config.method}.json").write_text(json.dumps(result, indent=2))
    return result


def _logits_and_labels(trainer: Trainer, dataset) -> tuple[np.ndarray, np.ndarray]:
    output = trainer.predict(dataset)
    return output.predictions, output.label_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--method", choices=["qlora", "full"], required=True)
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    result = run(Config(method=args.method, epochs=args.epochs), args.out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
