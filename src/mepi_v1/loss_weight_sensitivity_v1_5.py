"""Prepared MEPI v1.5 validation-only loss-weight sensitivity experiment.

Nothing runs on import. Every configuration starts independently from the same
selected MagNet representation with seed 42. Cross-lambda weighted totals are
never used to declare a winner because their objective definitions differ.
"""
from __future__ import annotations

import json, math
from pathlib import Path
from typing import Any

import torch
import yaml

from .checkpointing import save_checkpoint
from .config import load_config, resolve_path
from .finetune_notebook import set_reproducibility_seed
from .finetune_v1_4 import (
    StrictEarlyStopping, build_development_datasets, build_epoch_train_loader,
    build_validation_loader, capture_rng_state, compute_v1_4_losses,
    load_steinmetz_prior, make_amp_scaler, restore_rng_state,
    sha256_file, steinmetz_reference_z,
)
from .finetune_v1_5 import (
    CHECKPOINT_SHA256, MAX_EPOCHS, PROTOCOL_SHA256, PROTOCOL_VERSION,
    _atomic_json, construct_v1_5_model, make_optimizer_v1_5,
    validate_v1_5_config,
)

LAMBDA3 = (0.1, 0.3, 0.5, 1.0)
LAMBDA4 = (0.05, 0.10, 0.20)
PRIMARY_SELECTION_METRIC = "VAL_TASK_SCORE"
BASELINE_PAIR = (0.3, 0.20)


def load_sensitivity_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["_config_path"] = str(path)
    validate_sensitivity_config(config)
    return config


def grid_pairs() -> list[tuple[float, float]]:
    return [(l3, l4) for l3 in LAMBDA3 for l4 in LAMBDA4]


def validate_sensitivity_config(config: dict[str, Any]) -> None:
    if config.get("protocol_version") != PROTOCOL_VERSION or config.get("protocol_sha256") != PROTOCOL_SHA256:
        raise AssertionError("sensitivity config is not frozen to MEPI v1.5")
    weights = config.get("loss_weights", {})
    if weights != {"lambda1": [1.0], "lambda2": [1.0], "lambda3": list(LAMBDA3), "lambda4": list(LAMBDA4)}:
        raise AssertionError("loss-weight grid changed")
    training = config.get("training", {})
    expected = {"learning_rate": 5e-6, "weight_decay": 1e-4, "scheduler": None, "warmup": None, "batch_size": 64, "max_epochs": 100, "early_stopping_patience": 10, "min_delta": 0.0, "strict_improvement": True, "gradient_clip_norm": 1.0}
    if any(training.get(k) != v for k, v in expected.items()):
        raise AssertionError("sensitivity training contract changed")
    initialization = config.get("initialization", {})
    if initialization.get("checkpoint_sha256") != CHECKPOINT_SHA256 or initialization.get("reuse_other_finetuned_configuration") is not False:
        raise AssertionError("every grid run must initialize from the selected representation")
    if config.get("seed") != 42 or config.get("selection_rule_ready") is not True or config.get("automatic_winner_declaration") is not True or config.get("test_access") is not False:
        raise AssertionError("sensitivity safety contract changed")
    rule = config.get("selection_rule", {})
    if rule.get("primary_metric") != PRIMARY_SELECTION_METRIC or rule.get("formula") != "validation_L_electrical + validation_L_LSP" or rule.get("direction") != "minimize" or rule.get("weighted_validation_total_loss_cross_configuration_ranking") is not False or rule.get("exclude_numerical_failure") is not True:
        raise AssertionError("frozen validation-only selection rule changed")
    if len(grid_pairs()) != 12:
        raise AssertionError("expected exactly 12 configurations")


def configuration_id(lambda3: float, lambda4: float) -> str:
    return f"l3_{lambda3:g}_l4_{lambda4:.2f}".replace(".", "p")


def build_grid_plan(config_path: str | Path) -> list[dict[str, Any]]:
    config = load_sensitivity_config(config_path)
    checkpoint = config["initialization"]["checkpoint"]
    baseline_valid = config["baseline_reuse"]["valid"]
    return [
        {
            "configuration_id": configuration_id(l3, l4),
            "lambda1": 1.0,
            "lambda2": 1.0,
            "lambda3": l3,
            "lambda4": l4,
            "seed": 42,
            "initialization_checkpoint": checkpoint,
            "initialization_checkpoint_sha256": CHECKPOINT_SHA256,
            "max_epochs": 100,
            "patience": 10,
            "scheduler": None,
            "reuse_completed_baseline": baseline_valid and (l3, l4) == BASELINE_PAIR,
            "requires_new_run": not (baseline_valid and (l3, l4) == BASELINE_PAIR),
            "test_access": False,
        }
        for l3, l4 in grid_pairs()
    ]


def compute_val_task_score(validation_L_electrical: float, validation_L_LSP: float) -> float:
    values = (float(validation_L_electrical), float(validation_L_LSP))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("VAL_TASK_SCORE inputs must be finite")
    return values[0] + values[1]


def rank_completed_configurations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in rows if not row["numerical_failure"] and row["eligible_for_selection"]]
    if not eligible:
        return {"winner_declared": False, "selected_configuration_id": None, "tied_configuration_ids": [], "reason": "NO_NUMERICALLY_VALID_CONFIGURATION"}
    ranked = sorted(eligible, key=lambda row: row[PRIMARY_SELECTION_METRIC])
    minimum = ranked[0][PRIMARY_SELECTION_METRIC]
    tied = [row["configuration_id"] for row in ranked if row[PRIMARY_SELECTION_METRIC] == minimum]
    return {
        "winner_declared": len(tied) == 1,
        "selected_configuration_id": tied[0] if len(tied) == 1 else None,
        "tied_configuration_ids": tied,
        "minimum_VAL_TASK_SCORE": minimum,
        "reason": "UNIQUE_MINIMUM_VAL_TASK_SCORE" if len(tied) == 1 else "EXACT_PRIMARY_SCORE_TIE_REQUIRES_EXPLICIT_RESOLUTION",
    }


def _compatibility(lambda3: float, lambda4: float) -> dict[str, Any]:
    return {"experiment": "MEPI_V1_5_LOSS_WEIGHT_SENSITIVITY", "protocol_version": PROTOCOL_VERSION, "protocol_sha256": PROTOCOL_SHA256, "checkpoint_sha256": CHECKPOINT_SHA256, "optimizer": "AdamW", "learning_rate": 5e-6, "weight_decay": 1e-4, "scheduler": None, "lambda1": 1.0, "lambda2": 1.0, "lambda3": lambda3, "lambda4": lambda4, "seed": 42, "fresh_representation_initialization": True}


def _loss_epoch(model, loader, *, prior, device, lambda3, lambda4, optimizer=None, scaler=None, clip=1.0) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}; count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for raw in loader:
            batch = {k: v.to(device) for k, v in raw.items()}
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                p_st_z = steinmetz_reference_z(batch["frequency_hz"], batch["B_peak_t"], prior)
                outputs = model(batch["waveform"], batch["tabular"], p_st_z)
                losses = compute_v1_4_losses(outputs, batch, lambda3=lambda3, lambda4=lambda4)
            if training:
                scaler.scale(losses["total"]).backward(); scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                scaler.step(optimizer); scaler.update()
            size = len(batch["LSP_z"]); count += size
            for name, value in losses.items(): totals[name] = totals.get(name, 0.0) + float(value.detach().cpu()) * size
    return {name: value / count for name, value in totals.items()}


def _save(path, *, model, optimizer, scaler, epoch, stopping, compatibility, history) -> None:
    save_checkpoint({"format_version": "mepi-v1.5-loss-weight-resume-v1", "epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "best_validation_loss": stopping.best_validation_total_loss, "best_epoch": stopping.best_epoch, "patience_counter": stopping.patience_counter, "amp_scaler_state": scaler.state_dict(), "rng_state": capture_rng_state(), "compatibility": compatibility, "history": history}, path)


def _load(path, *, model, optimizer, scaler, compatibility):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = {"epoch", "model_state", "optimizer_state", "best_validation_loss", "best_epoch", "patience_counter", "amp_scaler_state", "rng_state", "compatibility", "history"}
    missing = sorted(required - checkpoint.keys())
    if missing or checkpoint.get("compatibility") != compatibility or "scheduler_state" in checkpoint:
        raise ValueError(f"incompatible sensitivity resume state: missing={missing}")
    model.load_state_dict(checkpoint["model_state"], strict=True); optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scaler.is_enabled(): scaler.load_state_dict(checkpoint["amp_scaler_state"])
    restore_rng_state(checkpoint["rng_state"])
    stop = StrictEarlyStopping(patience=10, min_delta=0.0, best_validation_total_loss=float(checkpoint["best_validation_loss"]), best_epoch=int(checkpoint["best_epoch"]), patience_counter=int(checkpoint["patience_counter"]))
    return int(checkpoint["epoch"]), stop, list(checkpoint["history"])


def _metrics(target: torch.Tensor, prediction: torch.Tensor) -> dict[str, float]:
    target, prediction = target.double(), prediction.double(); error = prediction - target
    return {"mae": float(error.abs().mean()), "rmse": float(error.square().mean().sqrt()), "r2": float(1 - error.square().sum() / (target - target.mean()).square().sum())}


def evaluate_best_validation(model, loader, dataset, *, prior, device, lambda3, lambda4) -> dict[str, Any]:
    model.eval()
    pred = {key: [] for key in ("efficiency_z", "P_loss_z", "mu_LSP_z")}
    true = {key: [] for key in ("efficiency_z", "P_loss_z", "LSP_z")}
    totals: dict[str, float] = {}
    count = 0
    numerical_failure = False
    with torch.no_grad():
        for raw in loader:
            batch = {key: value.to(device) for key, value in raw.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                p_st_z = steinmetz_reference_z(batch["frequency_hz"], batch["B_peak_t"], prior)
                outputs = model(batch["waveform"], batch["tabular"], p_st_z)
                losses = compute_v1_4_losses(outputs, batch, lambda3=lambda3, lambda4=lambda4)
            numerical_failure |= any(not bool(torch.isfinite(value).all()) for value in (*batch.values(), *outputs.values(), *losses.values()))
            size = len(batch["LSP_z"])
            count += size
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value) * size
            for key in pred:
                pred[key].append(outputs[key].float().cpu())
            true["efficiency_z"].append(batch["electrical_z"][:, 0].float().cpu())
            true["P_loss_z"].append(batch["electrical_z"][:, 1].float().cpu())
            true["LSP_z"].append(batch["LSP_z"].float().cpu())
    pred = {key: torch.cat(value) for key, value in pred.items()}
    true = {key: torch.cat(value) for key, value in true.items()}
    losses = {key: value / count for key, value in totals.items()}
    scalers = dataset.normalization["target_scalers"]
    def inverse(value, name):
        return value * float(scalers[name]["scale"][0]) + float(scalers[name]["mean"][0])
    efficiency_true, efficiency_pred = inverse(true["efficiency_z"], "efficiency_percent"), inverse(pred["efficiency_z"], "efficiency_percent")
    p_loss_true, p_loss_pred = inverse(true["P_loss_z"], "P_loss"), inverse(pred["P_loss_z"], "P_loss")
    lsp_true, lsp_pred = inverse(true["LSP_z"], "LSP_raw"), inverse(pred["mu_LSP_z"], "LSP_raw")
    if bool((lsp_true == 0).any()):
        raise ValueError("physical LSP MAPE undefined for zero target")
    physical = {
        "efficiency_percent": _metrics(efficiency_true, efficiency_pred),
        "P_loss": _metrics(p_loss_true, p_loss_pred),
        "LSP_raw": {**_metrics(lsp_true, lsp_pred), "mape_percent": float(((lsp_pred - lsp_true).abs() / lsp_true.abs()).mean() * 100)},
    }
    weighted = {"L_electrical": losses["electrical"], "L_LSP": losses["LSP"], "L_physics": lambda3 * losses["physics"], "L_UQ": lambda4 * losses["uncertainty_nll"]}
    required_numbers = [*losses.values(), *weighted.values(), *(metric for group in physical.values() for metric in group.values())]
    numerical_failure |= not all(math.isfinite(float(value)) for value in required_numbers)
    score = None if numerical_failure else compute_val_task_score(losses["electrical"], losses["LSP"])
    return {
        "sample_count": count,
        "physical_validation_metrics": physical,
        "Efficiency_MAE": physical["efficiency_percent"]["mae"],
        "Efficiency_RMSE": physical["efficiency_percent"]["rmse"],
        "Efficiency_R2": physical["efficiency_percent"]["r2"],
        "P_loss_MAE": physical["P_loss"]["mae"],
        "P_loss_RMSE": physical["P_loss"]["rmse"],
        "P_loss_R2": physical["P_loss"]["r2"],
        "LSP_raw_MAE": physical["LSP_raw"]["mae"],
        "LSP_raw_RMSE": physical["LSP_raw"]["rmse"],
        "LSP_raw_R2": physical["LSP_raw"]["r2"],
        "LSP_raw_MAPE_PERCENT": physical["LSP_raw"]["mape_percent"],
        "validation_L_electrical": losses["electrical"],
        "validation_L_LSP": losses["LSP"],
        "validation_L_physics": losses["physics"],
        "validation_L_UQ": losses["uncertainty_nll"],
        "unweighted_validation_losses": {"L_electrical": losses["electrical"], "L_LSP": losses["LSP"], "L_physics": losses["physics"], "L_UQ": losses["uncertainty_nll"]},
        "weighted_contributions": weighted,
        "weighted_validation_total_loss": losses["total"],
        PRIMARY_SELECTION_METRIC: score,
        "numerical_failure": numerical_failure,
        "eligible_for_selection": not numerical_failure,
        "test_accessed": False,
    }


def run_single_configuration(sensitivity_config_path: str | Path, *, lambda3: float, lambda4: float) -> dict[str, Any]:
    config=load_sensitivity_config(sensitivity_config_path)
    if (lambda3,lambda4) not in grid_pairs(): raise ValueError("pair is outside frozen grid")
    base_path=resolve_path(config,config["base_config"]);base=load_config(base_path);validate_v1_5_config(base)
    output_root=resolve_path(config,config["output_root"]);run=output_root/configuration_id(lambda3,lambda4);run.mkdir(parents=True,exist_ok=True)
    completed=run/"completed.json"
    if completed.exists(): return json.loads(completed.read_text())
    set_reproducibility_seed(42);train_ds,val_ds=build_development_datasets(base_path);val_loader=build_validation_loader(val_ds,batch_size=64);device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model,_=construct_v1_5_model(base_path);model.to(device);optimizer=make_optimizer_v1_5(model,base);scaler=make_amp_scaler(device);prior=load_steinmetz_prior(base);compat=_compatibility(lambda3,lambda4);last=run/"last_checkpoint.pt";best=run/"best_checkpoint.pt"
    start=0;history=[];stopping=StrictEarlyStopping(patience=10,min_delta=0.0)
    if last.exists(): start,stopping,history=_load(last,model=model,optimizer=optimizer,scaler=scaler,compatibility=compat)
    stopped=False
    for epoch in range(start,MAX_EPOCHS):
        train_loader=build_epoch_train_loader(train_ds,batch_size=64,seed=42,epoch=epoch);tl=_loss_epoch(model,train_loader,prior=prior,device=device,lambda3=lambda3,lambda4=lambda4,optimizer=optimizer,scaler=scaler);vl=_loss_epoch(model,val_loader,prior=prior,device=device,lambda3=lambda3,lambda4=lambda4)
        improved,should_stop=stopping.update(vl["total"],epoch);history.append({"epoch":epoch,"learning_rate":5e-6,"train":tl,"validation":vl,"strict_improvement":improved,"patience_counter":stopping.patience_counter})
        if improved:_save(best,model=model,optimizer=optimizer,scaler=scaler,epoch=epoch+1,stopping=stopping,compatibility=compat,history=history)
        _save(last,model=model,optimizer=optimizer,scaler=scaler,epoch=epoch+1,stopping=stopping,compatibility=compat,history=history);_atomic_json(run/"history.json",{"epochs":history})
        if should_stop:stopped=True;break
    ck=torch.load(best,map_location=device,weights_only=False);model.load_state_dict(ck["model_state"],strict=True);metrics=evaluate_best_validation(model,val_loader,val_ds,prior=prior,device=device,lambda3=lambda3,lambda4=lambda4)
    result={"configuration_id":configuration_id(lambda3,lambda4),"lambda1":1.0,"lambda2":1.0,"lambda3":lambda3,"lambda4":lambda4,"best_epoch":stopping.best_epoch,"stop_epoch":len(history)-1,"epochs_completed":len(history),"stopped_early":stopped,"initialization_checkpoint_sha256":CHECKPOINT_SHA256,"seed":42,"scheduler":None,**metrics,"selection_rule_ready":True,"primary_selection_metric":PRIMARY_SELECTION_METRIC}
    _atomic_json(completed,result);return result


def run_loss_weight_grid(config_path: str | Path) -> dict[str, Any]:
    config = load_sensitivity_config(config_path)
    if config.get("allow_grid_training") is not True:
        raise RuntimeError("grid training is disabled")
    rows = [run_single_configuration(config_path, lambda3=l3, lambda4=l4) for l3, l4 in grid_pairs()]
    selection = rank_completed_configurations(rows)
    payload = {
        "status": "GRID_COMPLETE_VALIDATION_ONLY",
        "configuration_count": len(rows),
        "comparison_table": rows,
        "selection_rule_ready": True,
        "primary_selection_metric": PRIMARY_SELECTION_METRIC,
        "weighted_validation_total_loss_used_for_cross_configuration_ranking": False,
        "selection": selection,
        "test_accessed": False,
    }
    output = resolve_path(config, config["output_root"])
    _atomic_json(output / "comparison.json", payload)
    return payload
