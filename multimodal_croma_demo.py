"""CLI entry point for paired radar-optical CROMA split-federated training."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import time

import torch
from torch import nn

from croma_models import (
    ContrastiveProjectionHead,
    FeatureProjection,
    PatchSegmentationHead,
    build_croma_components,
    configure_satellite_encoder_trainability,
    pretrained_encoder_schedule,
)
from multimodal_data import (
    PairedBatchSequence,
    build_dataset_bundle,
    partition_dataset_indices,
)
from multimodal_evaluation import evaluate_global, write_csv
from multimodal_sfl import (
    ModalityState, PairContribution, PlanePair, average_states, clone_state,
    estimate_transaction, make_ema_teacher, reset_pair_from_global, train_pair_offline,
    train_server_on_matched_features,
)
from orbit_model import parse_utc
from pair_contact_scheduler import build_pair_contacts, load_raw_contacts, utc_at


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_RUNS_DIR = PROJECT_DIR.parent / "multimodal_croma_demo"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {requested!r} was requested but is unavailable")
    return device


def create_timestamped_run_dir(base_dir: Path, config_path: Path) -> Path:
    """Create one immutable output directory and preserve the input config."""
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = base_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, run_dir / "config.json")
    return run_dir


def make_plane_pairs(pair_contacts, global_states, config, dataset_bundle):
    training = config["segmentation_training"]
    pair_ids_by_plane = {
        plane: next(row["pair_id"] for row in pair_contacts if row["plane"] == plane)
        for plane in sorted({row["plane"] for row in pair_contacts})
    }
    pair_indices = partition_dataset_indices(
        len(dataset_bundle.train), list(pair_ids_by_plane.values()), int(training["seed"])
    )
    pairs = {}
    for plane, pair_id in pair_ids_by_plane.items():
        example = next(row for row in pair_contacts if row["plane"] == plane)
        pairs[example["pair_id"]] = PlanePair(
            pair_id=example["pair_id"], plane=plane,
            radar=ModalityState(
                example["radar_satellite_id"], "radar",
                {key: value.clone() for key, value in global_states["radar_encoder"].items()},
                {key: value.clone() for key, value in global_states["radar_auxiliary"].items()},
                {key: value.clone() for key, value in global_states["radar_projection"].items()},
                contrastive_state={key: value.clone() for key, value in global_states["radar_contrastive"].items()},
            ),
            optical=ModalityState(
                example["optical_satellite_id"], "optical",
                {key: value.clone() for key, value in global_states["optical_encoder"].items()},
                {key: value.clone() for key, value in global_states["optical_auxiliary"].items()},
                {key: value.clone() for key, value in global_states["optical_projection"].items()},
                contrastive_state={key: value.clone() for key, value in global_states["optical_contrastive"].items()},
            ),
            batches=PairedBatchSequence(
                dataset_bundle.train,
                pair_indices[pair_id],
                training["batch_size"],
                training["epochs"],
                training["seed"] + plane,
            ),
        )
    return pairs


def run_training(config, raw_contacts, output_dir: Path):
    """运行配对多模态 CROMA 拆分联邦学习(SFL)仿真的主流程。

    以地面可见窗口为时间轴:每个窗口内先让卫星对完成不可见时段的星上本地
    训练,过站时上传匹配特征供地面端融合训练;每攒满 aggregation_k 次贡献
    做一次全局聚合。结束后评估、写日志并保存 checkpoint。
    """
    run_started_at = time.perf_counter()
    training, model_config = config["segmentation_training"], config["croma"]
    torch.manual_seed(training["seed"])
    torch.set_num_threads(1)
    device = select_device(training["device"])
    # 仿真时间轴:epoch 为零点,horizon_s 为总时长(秒)
    epoch = parse_utc(config["simulation"]["epoch_utc"])
    horizon_s = float(config["simulation"]["duration_hours"]) * 3600.0
    # 把单星直接可见窗口按轨道面合并成配对窗口(至少一颗星可见的并集区间)
    pair_contacts = build_pair_contacts(raw_contacts, epoch)
    pair_count = len({row["pair_id"] for row in pair_contacts})
    dataset_bundle = build_dataset_bundle(config, PROJECT_DIR, pair_count)
    config["dataset_metadata"] = {
        "name": dataset_bundle.metadata.name,
        "ignore_index": dataset_bundle.metadata.ignore_index,
        "num_classes": dataset_bundle.metadata.num_classes,
        "radar_channels": dataset_bundle.metadata.radar_channels,
        "optical_channels": dataset_bundle.metadata.optical_channels,
    }
    num_classes = dataset_bundle.metadata.num_classes
    image_size = dataset_bundle.metadata.image_size
    # 模型组件:编码器模块全局共享,各卫星对通过换入/换出自己的 state_dict 区分(单机模拟多星)
    radar_worker, optical_worker, cross_encoder, attention_bias, checkpoint_status = build_croma_components(config, device, PROJECT_DIR)
    ema_teacher_enabled = bool(training.get("ema_teacher_enabled", True))
    ema_teacher_decay = float(training.get("ema_teacher_decay", 0.99))
    ema_teacher_decay_start = float(training.get("ema_teacher_decay_start", ema_teacher_decay))
    ema_teacher_anneal_steps = max(0, int(training.get("ema_teacher_anneal_steps", 0)))
    distillation_weight_start = float(training.get("distillation_weight_start", 1.0))
    distillation_weight_end = float(training.get("distillation_weight_end", 1.0))
    distillation_anneal_steps = max(0, int(training.get("distillation_anneal_steps", 0)))
    if ema_teacher_enabled and not all(0.0 <= value < 1.0 for value in (ema_teacher_decay_start, ema_teacher_decay)):
        raise ValueError("ema_teacher_decay_start and ema_teacher_decay must be in the interval [0, 1)")
    if distillation_weight_start < 0.0 or distillation_weight_end < 0.0:
        raise ValueError("distillation weights must be non-negative")
    cross_encoder_ema_teacher = make_ema_teacher(cross_encoder) if ema_teacher_enabled else None
    # 星上使用一份共享辅助头:雷达和光学 encoder 的输出依次更新同一组参数。
    # 保留两个变量名是为了兼容现有训练接口,它们明确指向同一个模块。
    radar_auxiliary = PatchSegmentationHead(model_config["encoder_dim"], num_classes, model_config["num_patches"]).to(device)
    optical_auxiliary = radar_auxiliary
    ground_head = PatchSegmentationHead(model_config["encoder_dim"], num_classes, model_config["num_patches"]).to(device)
    radar_projection = FeatureProjection(model_config["encoder_dim"]).to(device)
    optical_projection = FeatureProjection(model_config["encoder_dim"]).to(device)
    contrastive_learning_enabled = bool(training.get("contrastive_learning_enabled", True))
    contrastive_projection_dim = int(training.get("contrastive_projection_dim", model_config["encoder_dim"]))
    contrastive_hidden_dim = int(training.get("contrastive_hidden_dim", model_config["encoder_dim"]))
    if contrastive_learning_enabled:
        radar_contrastive = ContrastiveProjectionHead(
            model_config["encoder_dim"], contrastive_projection_dim, contrastive_hidden_dim
        ).to(device)
        optical_contrastive = ContrastiveProjectionHead(
            model_config["encoder_dim"], contrastive_projection_dim, contrastive_hidden_dim
        ).to(device)
    else:
        radar_contrastive = optical_contrastive = None
    criterion = nn.CrossEntropyLoss(ignore_index=dataset_bundle.metadata.ignore_index)
    # 地面端训练 cross encoder、地面头和两个特征投影层；投影层随后下发给对应卫星。
    server_optimizer = torch.optim.AdamW(
        list(cross_encoder.parameters()) + list(ground_head.parameters())
        + list(radar_projection.parameters()) + list(optical_projection.parameters()),
        lr=training["server_learning_rate"],
        weight_decay=float(training.get("weight_decay", 0.01)),
    )
    # 全局模型状态(FedAvg 聚合对象):各卫星对由它初始化,聚合后重置回它
    global_states = {
        "radar_encoder": clone_state(radar_worker), "radar_auxiliary": clone_state(radar_auxiliary),
        "radar_projection": clone_state(radar_projection),
        "radar_contrastive": clone_state(radar_contrastive) if radar_contrastive is not None else {},
        "optical_encoder": clone_state(optical_worker), "optical_auxiliary": clone_state(optical_auxiliary),
        "optical_projection": clone_state(optical_projection),
        "optical_contrastive": clone_state(optical_contrastive) if optical_contrastive is not None else {},
    }
    # 每个轨道面构建一个卫星对:本地状态副本 + 确定性批次流
    pairs = make_plane_pairs(pair_contacts, global_states, config, dataset_bundle)
    test_data = dataset_bundle.validation
    encoder_schedule = pretrained_encoder_schedule(config)
    encoder_stage = configure_satellite_encoder_trainability(
        radar_worker, optical_worker, config, global_version=0
    )
    if encoder_schedule["mode"] != "full":
        print(
            "[pretrained-encoder] "
            f"mode={encoder_schedule['mode']} "
            f"stage={encoder_stage} "
            f"warmup_aggregations={encoder_schedule['warmup_aggregations']} "
            f"trainable_blocks={encoder_schedule['trainable_blocks']}",
            flush=True,
        )
    # 每攒满 k 个卫星对的贡献做一次全局聚合
    aggregation_k = int(training["aggregation_k"])
    if not 1 <= aggregation_k <= len(pairs):
        raise ValueError("aggregation_k must be between 1 and the plane count")
    # pending:待聚合的贡献缓冲;三个日志分别记录本地训练 / 过站事务 / 聚合事件
    pending, local_log, contact_log, aggregation_log = [], [], [], []
    # ground_available_s:地面站最早可用时刻(串行独占资源)
    # global_version:全局模型版本号;server_updates:服务器累计训练步数
    ground_available_s, global_version, server_updates = 0.0, 0, 0
    # 离散事件主循环:按时间顺序处理每个配对可见窗口
    for contact in pair_contacts:
        pair: PlanePair = pairs[contact["pair_id"]]
        # 单卡用共享 worker 模拟多个卫星；每个窗口前应用当前冻结阶段。
        configure_satellite_encoder_trainability(
            radar_worker, optical_worker, config, global_version
        )
        # 1) 窗口开始前:卫星对在不可见时段做星上本地训练,产出带版本号的特征包
        train_pair_offline(
            pair, contact["start_offset_s"], config, radar_worker, optical_worker,
            radar_auxiliary, optical_auxiliary, radar_projection, optical_projection,
            attention_bias, criterion, device, epoch, local_log,
            radar_contrastive, optical_contrastive,
        )
        # 2) 找到双模态缓冲区中可对齐融合的批次(batch_number 交集)
        matched_ids = pair.matched_batch_ids()
        # 本次事务是否会触发聚合(聚合耗时需计入事务时长估算)
        will_aggregate = len(pending) + 1 >= aggregation_k
        # 估算本次过站事务的传输字节数与总耗时(上行 + 传播 + 服务器计算 + 聚合 + 下行)
        estimate = estimate_transaction(pair, matched_ids, contact, config, will_aggregate)
        # 3) 排期:最早只能在窗口开始且地面站空闲后开始;截止时间预留安全余量
        start_s = max(contact["start_offset_s"], ground_available_s)
        finish_s = start_s + estimate["duration_s"]
        deadline_s = contact["end_offset_s"] - float(config["link"]["safety_margin_s"])
        status, reason = "completed", ""
        losses = {
            "segmentation": [], "radar_distillation": [], "optical_distillation": [],
            "ema_teacher_decay": [], "distillation_weight": [],
        }
        aggregation_members, aggregation_performed = "", False
        test_accuracy, test_miou = "", ""
        server_mean_loss = ""
        # 4) 三种跳过情形:无匹配特征 / 地面站忙到窗口断开 / 事务放不进窗口
        if not matched_ids:
            status, reason = "skipped", "no_matched_multimodal_features"
        elif start_s >= contact["end_offset_s"]:
            status, reason = "skipped", "ground_station_busy_until_disconnect"
        elif finish_s > deadline_s:
            status, reason = "skipped", "paired_transaction_does_not_fit_contact"
        else:
            # 5) 地面端:跨模态编码器 + 地面头在匹配特征上做监督训练
            losses = train_server_on_matched_features(
                pair, matched_ids, cross_encoder, ground_head, radar_projection,
                optical_projection, attention_bias, server_optimizer, criterion,
                image_size, device, cross_encoder_ema_teacher, ema_teacher_decay,
                ema_teacher_decay_start, ema_teacher_anneal_steps,
                distillation_weight_start, distillation_weight_end,
                distillation_anneal_steps, server_updates,
            )
            server_updates += len(losses["segmentation"])
            server_mean_loss = (
                sum(losses["segmentation"]) / len(losses["segmentation"])
                if losses["segmentation"] else ""
            )
            global_states["radar_projection"] = clone_state(radar_projection)
            global_states["optical_projection"] = clone_state(optical_projection)
            # 该对上传的四份状态(雷达/光学编码器 + 辅助头)作为一次待聚合贡献
            pending.append(PairContribution(
                pair.pair_id, pair.radar.encoder_state, pair.radar.auxiliary_state,
                pair.optical.encoder_state, pair.optical.auxiliary_state,
                pair.radar.contrastive_state, pair.optical.contrastive_state,
            ))
            if len(pending) == aggregation_k:
                # 6) 凑满 k 个贡献:按模态做均匀算术平均(FedAvg),产生新的全局版本
                aggregation_members = ";".join(item.pair_id for item in pending)
                global_states = {
                    "radar_encoder": average_states([item.radar_encoder_state for item in pending]),
                    "radar_auxiliary": average_states([item.radar_auxiliary_state for item in pending]),
                    "radar_projection": global_states["radar_projection"],
                    "radar_contrastive": average_states([item.radar_contrastive_state for item in pending]),
                    "optical_encoder": average_states([item.optical_encoder_state for item in pending]),
                    "optical_auxiliary": average_states([item.optical_auxiliary_state for item in pending]),
                    "optical_projection": global_states["optical_projection"],
                    "optical_contrastive": average_states([item.optical_contrastive_state for item in pending]),
                }
                global_version, aggregation_performed = global_version + 1, True
                encoder_stage = configure_satellite_encoder_trainability(
                    radar_worker, optical_worker, config, global_version
                )
                aggregation_log.append({"global_version": global_version, "finish_utc": utc_at(epoch, finish_s), "plane_pairs": aggregation_members, "pair_count": aggregation_k, "weight_per_pair": round(1.0 / aggregation_k, 8)})
                # Evaluate the newly aggregated global model once per aggregation.
                test_accuracy, test_miou, _ = evaluate_global(
                    test_data, radar_worker, optical_worker, cross_encoder,
                    ground_head, attention_bias, global_states, config, device,
                )
                elapsed_s = time.perf_counter() - run_started_at
                runtime_text = f"[{int(elapsed_s // 60):02d}:{int(elapsed_s % 60):02d}]"
                server_loss_text = (
                    f"{server_mean_loss:.8f}"
                    if server_mean_loss != "" else "n/a"
                )
                print(
                    "[aggregation] "
                    f"window_end_utc={contact['end_utc']} "
                    f"runtime={runtime_text} "
                    f"server_mean_loss={server_loss_text} "
                    f"test_accuracy={test_accuracy:.8f} test_miou={test_miou:.8f}",
                    flush=True,
                )
                pending.clear()
            # 7) 该对重置到最新全局模型,并清空特征缓冲(避免上传陈旧特征)
            reset_pair_from_global(pair, global_states, global_version)
            pair.radar_buffer.clear()
            pair.optical_buffer.clear()
            # 地面站被本次事务占用,直到事务结束才空闲
            ground_available_s = finish_s
        # 8) 无论事务完成还是跳过,都记录本次窗口的结果(原因、字节数、服务器损失、聚合信息等)
        contact_log.append({
            "pair_contact_id": contact["pair_contact_id"], "pair_id": pair.pair_id,
            "radar_satellite_id": pair.radar.satellite_id, "optical_satellite_id": pair.optical.satellite_id,
            "direct_satellites": contact["direct_satellites"], "contact_start_utc": contact["start_utc"], "contact_end_utc": contact["end_utc"],
            "transaction_start_utc": utc_at(epoch, start_s) if status == "completed" else "",
            "transaction_finish_utc": utc_at(epoch, finish_s) if status == "completed" else "",
            "status": status, "reason": reason, "matched_batches": len(matched_ids),
            "radar_upload_bytes": estimate["radar_upload_bytes"], "optical_upload_bytes": estimate["optical_upload_bytes"],
            "server_updates": len(losses["segmentation"]),
            "server_mean_loss": round(server_mean_loss, 8) if server_mean_loss != "" else "",
            "radar_distillation_mean_loss": round(sum(losses["radar_distillation"]) / len(losses["radar_distillation"]), 8) if losses["radar_distillation"] else "",
            "optical_distillation_mean_loss": round(sum(losses["optical_distillation"]) / len(losses["optical_distillation"]), 8) if losses["optical_distillation"] else "",
            "ema_teacher_decay": round(sum(losses["ema_teacher_decay"]) / len(losses["ema_teacher_decay"]), 8) if losses["ema_teacher_decay"] else "",
            "distillation_weight": round(sum(losses["distillation_weight"]) / len(losses["distillation_weight"]), 8) if losses["distillation_weight"] else "",
            "test_accuracy": test_accuracy, "test_miou": test_miou,
            "aggregation_performed": int(aggregation_performed), "aggregation_members": aggregation_members,
            "global_version": global_version, "modeled_transaction_s": round(estimate["duration_s"], 8),
        })
        # 星上本地训练只允许发生在不可见时段:把该对时钟直接推进到窗口结束
        pair.local_clock_s = max(pair.local_clock_s, contact["end_offset_s"])

    # 主循环结束后:各对在剩余时间内完成剩余的本地训练(不再有新事务与聚合)
    for pair in pairs.values():
        train_pair_offline(
            pair, horizon_s, config, radar_worker, optical_worker,
            radar_auxiliary, optical_auxiliary, radar_projection, optical_projection,
            attention_bias, criterion, device, epoch, local_log,
            radar_contrastive, optical_contrastive,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    # 输出全部日志:配对窗口 / 本地训练 / 过站事务 / 聚合事件
    write_csv(output_dir / "pair_contact_windows.csv", pair_contacts)
    local_log.sort(key=lambda row: (row["start_utc"], row["pair_id"]))
    write_csv(output_dir / "multimodal_local_training_log.csv", local_log)
    write_csv(output_dir / "multimodal_training_log.csv", contact_log)
    write_csv(output_dir / "multimodal_aggregation_log.csv", aggregation_log, ["global_version", "finish_utc", "plane_pairs", "pair_count", "weight_per_pair"])
    # 用全局模型状态在测试集上评估:像素精度、mIoU、混淆矩阵
    pixel_accuracy, mean_iou, confusion = evaluate_global(test_data, radar_worker, optical_worker, cross_encoder, ground_head, attention_bias, global_states, config, device)
    # 统计事务成功率(completed 与 skipped 数量)
    successful = sum(row["status"] == "completed" for row in contact_log)
    # 汇总统计信息(规模、成功率、聚合次数、精度指标等)
    summary = {
        "algorithm": "paired multimodal CROMA SFL with ground feature distillation and without staleness weighting", "device": str(device),
        "croma_profile": model_config["profile"], "checkpoint": checkpoint_status, "image_size": image_size,
        "dataset": dataset_bundle.metadata.name,
        "train_samples": len(dataset_bundle.train),
        "validation_samples": len(dataset_bundle.validation),
        "samples_per_pair": {
            pair_id: len({index for _, _, indices in pair.batches.batch_specs for index in indices})
            for pair_id, pair in pairs.items()
        },
        "plane_pairs": len(pairs), "pair_contact_windows": len(pair_contacts), "local_paired_steps": len(local_log),
        "server_updates": server_updates, "successful_pair_transactions": successful,
        "skipped_pair_contacts": len(contact_log) - successful, "aggregations": len(aggregation_log),
        "aggregation_k": aggregation_k, "aggregation": "uniform arithmetic mean per modality",
        "pretrained_encoder_schedule": {
            "mode": str(encoder_schedule["mode"]),
            "warmup_aggregations": int(encoder_schedule["warmup_aggregations"]),
            "trainable_blocks_after_warmup": int(encoder_schedule["trainable_blocks"]),
            "final_stage": encoder_stage,
        },
        "projection_distillation": "projR/projO MSE to detached EMA cross_encoder features",
        "ema_teacher": {
            "enabled": ema_teacher_enabled,
            "decay_start": ema_teacher_decay_start if ema_teacher_enabled else None,
            "decay_end": ema_teacher_decay if ema_teacher_enabled else None,
            "anneal_steps": ema_teacher_anneal_steps if ema_teacher_enabled else None,
            "distillation_weight_start": distillation_weight_start,
            "distillation_weight_end": distillation_weight_end,
            "distillation_anneal_steps": distillation_anneal_steps,
            "target": "cross_encoder EMA output" if ema_teacher_enabled else "detached student output",
        },
        "satellite_auxiliary_head": "one shared two-layer convolutional head for radar and optical branches",
        "contrastive_learning": {
            "enabled": contrastive_learning_enabled,
            "loss_weight": float(training.get("contrastive_loss_weight", 0.1)),
            "temperature": float(training.get("contrastive_temperature", 0.07)),
            "projection_dim": contrastive_projection_dim,
            "hidden_dim": contrastive_hidden_dim,
            "objective": "symmetric in-batch InfoNCE over ISL-paired radar/optical embeddings",
        },
        "pixel_accuracy": pixel_accuracy, "mean_iou": mean_iou, 
        # "confusion_matrix": confusion,
        "last_ground_transaction_utc": utc_at(epoch, ground_available_s),
        "pending_matched_batches": {pair_id: len(pair.matched_batch_ids()) for pair_id, pair in pairs.items() if pair.matched_batch_ids()},
    }
    with (output_dir / "multimodal_training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    # 保存最终检查点:全局模型状态 + 服务器端跨模态编码器与地面头
    torch.save({
        "config": config, "global_version": global_version, "global_states": global_states,
        "cross_encoder_state": clone_state(cross_encoder),
        "cross_encoder_ema_teacher_state": clone_state(cross_encoder_ema_teacher) if cross_encoder_ema_teacher is not None else None,
        "ground_segmentation_head_state": clone_state(ground_head),
        "radar_projection_state": clone_state(radar_projection),
        "optical_projection_state": clone_state(optical_projection),
        "radar_contrastive_state": clone_state(radar_contrastive) if radar_contrastive is not None else None,
        "optical_contrastive_state": clone_state(optical_contrastive) if optical_contrastive is not None else None,
    }, output_dir / "multimodal_final_checkpoint.pt")
    print(json.dumps(summary, indent=2))
    print(f"Pair contacts: {output_dir / 'pair_contact_windows.csv'}")
    print(f"Training log: {output_dir / 'multimodal_training_log.csv'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "config.json")
    parser.add_argument("--contacts", type=Path, default=PROJECT_DIR / "outputs" / "contact_windows.csv")
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_RUNS_DIR,
        help="Base directory; each invocation creates a timestamped child directory.",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    run_dir = create_timestamped_run_dir(args.output_dir, config_path)
    run_training(load_json(config_path), load_raw_contacts(args.contacts), run_dir)
    print(f"Run output directory: {run_dir}")


if __name__ == "__main__":
    main()
