"""Train RADAR on a prepared patient-sequence dataset."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from .data import RadarDataset, collate_patient_sequences, move_batch
from .losses import competing_risk_nll, domain_loss, domain_schedule
from .model import RadarConfig, RadarModel, cumulative_incidence


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_loader(
    dataset: RadarDataset,
    batch_patients: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_patients,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_patient_sequences,
        generator=generator,
    )


def batch_loss(
    model: RadarModel,
    batch: dict[str, torch.Tensor],
    *,
    coefficient: float,
    domain_weight: float,
    return_scores: bool = False,
) -> (
    tuple[torch.Tensor, float, float] | tuple[torch.Tensor, float, float, torch.Tensor]
):
    output = model(
        code_ids=batch["code_ids"],
        code_offsets=batch["code_offsets"],
        patient_ptr=batch["patient_ptr"],
        event_days=batch["event_days"],
        ages=batch["ages"],
        sexes=batch["sexes"],
        diagnosis_counts=batch["diagnosis_counts"],
        domain_coefficient=coefficient,
    )
    clinical = competing_risk_nll(
        output.raw_hazards,
        batch["followup_days"],
        batch["event_types"],
        batch["risk_masks"],
    )
    domain = domain_loss(output.domain_logits, batch["coding_domains"])
    total = clinical + domain_weight * domain
    losses = total, float(clinical.detach()), float(domain.detach())
    if return_scores:
        return (*losses, cumulative_incidence(output.raw_hazards)[0][:, -1])
    return losses


@torch.no_grad()
def evaluate(
    model: RadarModel,
    loader: DataLoader,
    device: torch.device,
    domain_weight: float,
    max_batches: int,
) -> dict[str, float]:
    model.eval()
    totals = np.zeros(3, dtype=np.float64)
    n_batches = 0
    labels: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    for batch_index, batch in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        batch = move_batch(batch, device)
        total, clinical, domain, af_scores = batch_loss(
            model,
            batch,
            coefficient=0.0,
            domain_weight=domain_weight,
            return_scores=True,
        )
        valid = batch["label_masks_365d"]
        labels.append(batch["labels_365d"][valid].cpu().numpy())
        scores.append(af_scores[valid].cpu().numpy())
        totals += (float(total), clinical, domain)
        n_batches += 1
    if not n_batches:
        raise RuntimeError("validation loader produced no batches")
    y = np.concatenate(labels)
    if len(np.unique(y)) != 2:
        raise RuntimeError(
            "12-month validation AUROC requires both cases and controls with "
            "observable outcomes; use a representative validation split and "
            "avoid limiting validation to batches containing only one class"
        )
    return {
        "loss": float(totals[0] / n_batches),
        "clinical_nll": float(totals[1] / n_batches),
        "domain_bce": float(totals[2] / n_batches),
        "auroc_365d": float(roc_auc_score(y, np.concatenate(scores))),
    }


def learning_rate_multiplier(
    step: int, total_steps: int, warmup_fraction: float
) -> float:
    """Study schedule: linear warm-up followed by cosine decay."""
    warmup_steps = int(total_steps * warmup_fraction)
    if warmup_steps > 0 and step < warmup_steps:
        return max(step, 1) / warmup_steps
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-patients", type=int, default=16)
    parser.add_argument("--embedding-dim", type=int, default=96)
    parser.add_argument("--visit-queries", type=int, default=2)
    parser.add_argument("--memory-channels", type=int, default=8)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--domain-hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--half-lives-days",
        nargs="+",
        type=float,
        default=[7, 30, 90, 180, 365, 1095, 3650, 36500],
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-fraction", type=float, default=0.05)
    parser.add_argument("--domain-loss-weight", type=float, default=1.0)
    parser.add_argument("--domain-max-coefficient", type=float, default=0.03)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-validation-batches", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not 0 <= args.warmup_fraction < 1:
        raise ValueError("warmup fraction must lie in [0, 1)")
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.out_dir.exists():
        raise FileExistsError(f"{args.out_dir} already exists")
    args.out_dir.mkdir(parents=True)
    set_seed(args.seed)
    device = resolve_device(args.device)

    train_data = RadarDataset(args.data_dir, "train")
    validation_data = RadarDataset(args.data_dir, "validation")
    vocab_size = int(train_data.metadata["vocab_size"])
    config = RadarConfig(
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        visit_queries=args.visit_queries,
        memory_channels=args.memory_channels,
        attention_heads=args.attention_heads,
        domain_hidden_dim=args.domain_hidden_dim,
        dropout=args.dropout,
        half_lives_days=tuple(args.half_lives_days),
    )
    model = RadarModel(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    train_loader = build_loader(
        train_data, args.batch_patients, shuffle=True, seed=args.seed
    )
    validation_loader = build_loader(
        validation_data, args.batch_patients, shuffle=False, seed=args.seed
    )
    steps_per_epoch = min(
        len(train_loader), args.max_train_batches or len(train_loader)
    )
    total_steps = max(1, args.epochs * steps_per_epoch)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: learning_rate_multiplier(
            step, total_steps, args.warmup_fraction
        ),
    )
    history: list[dict] = []
    best_auc = -float("inf")
    global_step = 0
    started = time.time()

    print(
        f"device={device}; train_patients={len(train_data):,}; "
        f"validation_patients={len(validation_data):,}; vocab={vocab_size:,}"
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = np.zeros(3, dtype=np.float64)
        n_batches = 0
        for batch_index, batch in enumerate(train_loader):
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break
            batch = move_batch(batch, device)
            coefficient = domain_schedule(
                global_step, total_steps, args.domain_max_coefficient
            )
            optimizer.zero_grad(set_to_none=True)
            total, clinical, domain = batch_loss(
                model,
                batch,
                coefficient=coefficient,
                domain_weight=args.domain_loss_weight,
            )
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            scheduler.step()
            running += (float(total.detach()), clinical, domain)
            n_batches += 1
            global_step += 1

        validation = evaluate(
            model,
            validation_loader,
            device,
            args.domain_loss_weight,
            args.max_validation_batches,
        )
        row = {
            "epoch": epoch,
            "train_loss": float(running[0] / max(n_batches, 1)),
            "train_clinical_nll": float(running[1] / max(n_batches, 1)),
            "train_domain_bce": float(running[2] / max(n_batches, 1)),
            **{f"validation_{key}": value for key, value in validation.items()},
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True))

        payload = {
            "format_version": 1,
            "epoch": epoch,
            "model_config": config.to_dict(),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "selection_metric": "validation_auroc_365d",
            "validation": validation,
            "dataset_metadata": train_data.metadata,
        }
        torch.save(payload, args.out_dir / "checkpoint_last.pt")
        if validation["auroc_365d"] > best_auc:
            best_auc = validation["auroc_365d"]
            torch.save(payload, args.out_dir / "checkpoint_best.pt")

    (args.out_dir / "training_history.json").write_text(
        json.dumps(history, indent=2) + "\n"
    )
    (args.out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "arguments": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
                "model": config.to_dict(),
                "device": str(device),
                "elapsed_seconds": time.time() - started,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
