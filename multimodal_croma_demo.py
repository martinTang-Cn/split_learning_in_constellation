"""CLI entry point for paired radar-optical CROMA split-federated training."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil

import torch
from torch import nn

from croma_models import FeatureProjection, PatchSegmentationHead, build_croma_components
from multimodal_data import (
    PairedBatchSequence,
    build_dataset_bundle,
    partition_dataset_indices,
)
from multimodal_evaluation import evaluate_global, write_csv
from multimodal_sfl import (
    ModalityState, PairContribution, PlanePair, average_states, clone_state,
    estimate_transaction, reset_pair_from_global, train_pair_offline,
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
            ),
            optical=ModalityState(
                example["optical_satellite_id"], "optical",
                {key: value.clone() for key, value in global_states["optical_encoder"].items()},
                {key: value.clone() for key, value in global_states["optical_auxiliary"].items()},
                {key: value.clone() for key, value in global_states["optical_projection"].items()},
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
    # 星上本地训练用的辅助分割头(雷达、光学)与地面分割头
    radar_auxiliary = PatchSegmentationHead(model_config["encoder_dim"], num_classes, model_config["num_patches"]).to(device)
    optical_auxiliary = PatchSegmentationHead(model_config["encoder_dim"], num_classes, model_config["num_patches"]).to(device)
    ground_head = PatchSegmentationHead(model_config["encoder_dim"], num_classes, model_config["num_patches"]).to(device)
    radar_projection = FeatureProjection(model_config["encoder_dim"]).to(device)
    optical_projection = FeatureProjection(model_config["encoder_dim"]).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset_bundle.metadata.ignore_index)
    # 地面端训练 cross encoder、地面头和两个特征投影层；投影层随后下发给对应卫星。
    server_optimizer = torch.optim.SGD(
        list(cross_encoder.parameters()) + list(ground_head.parameters())
        + list(radar_projection.parameters()) + list(optical_projection.parameters()),
        lr=training["server_learning_rate"],
    )
    # 全局模型状态(FedAvg 聚合对象):各卫星对由它初始化,聚合后重置回它
    global_states = {
        "radar_encoder": clone_state(radar_worker), "radar_auxiliary": clone_state(radar_auxiliary),
        "radar_projection": clone_state(radar_projection),
        "optical_encoder": clone_state(optical_worker), "optical_auxiliary": clone_state(optical_auxiliary),
        "optical_projection": clone_state(optical_projection),
    }
    # 每个轨道面构建一个卫星对:本地状态副本 + 确定性批次流
    pairs = make_plane_pairs(pair_contacts, global_states, config, dataset_bundle)
    test_data = dataset_bundle.validation
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
        # 1) 窗口开始前:卫星对在不可见时段做星上本地训练,产出带版本号的特征包
        train_pair_offline(pair, contact["start_offset_s"], config, radar_worker, optical_worker, radar_auxiliary, optical_auxiliary, radar_projection, optical_projection, attention_bias, criterion, device, epoch, local_log)
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
        losses = {"segmentation": [], "radar_distillation": [], "optical_distillation": []}
        aggregation_members, aggregation_performed = "", False
        # 4) 三种跳过情形:无匹配特征 / 地面站忙到窗口断开 / 事务放不进窗口
        if not matched_ids:
            status, reason = "skipped", "no_matched_multimodal_features"
        elif start_s >= contact["end_offset_s"]:
            status, reason = "skipped", "ground_station_busy_until_disconnect"
        elif finish_s > deadline_s:
            status, reason = "skipped", "paired_transaction_does_not_fit_contact"
        else:
            # 5) 地面端:跨模态编码器 + 地面头在匹配特征上做监督训练
            losses = train_server_on_matched_features(pair, matched_ids, cross_encoder, ground_head, radar_projection, optical_projection, attention_bias, server_optimizer, criterion, image_size, device)
            server_updates += len(losses["segmentation"])
            global_states["radar_projection"] = clone_state(radar_projection)
            global_states["optical_projection"] = clone_state(optical_projection)
            # 该对上传的四份状态(雷达/光学编码器 + 辅助头)作为一次待聚合贡献
            pending.append(PairContribution(pair.pair_id, pair.radar.encoder_state, pair.radar.auxiliary_state, pair.optical.encoder_state, pair.optical.auxiliary_state))
            if len(pending) == aggregation_k:
                # 6) 凑满 k 个贡献:按模态做均匀算术平均(FedAvg),产生新的全局版本
                aggregation_members = ";".join(item.pair_id for item in pending)
                global_states = {
                    "radar_encoder": average_states([item.radar_encoder_state for item in pending]),
                    "radar_auxiliary": average_states([item.radar_auxiliary_state for item in pending]),
                    "radar_projection": global_states["radar_projection"],
                    "optical_encoder": average_states([item.optical_encoder_state for item in pending]),
                    "optical_auxiliary": average_states([item.optical_auxiliary_state for item in pending]),
                    "optical_projection": global_states["optical_projection"],
                }
                global_version, aggregation_performed = global_version + 1, True
                aggregation_log.append({"global_version": global_version, "finish_utc": utc_at(epoch, finish_s), "plane_pairs": aggregation_members, "pair_count": aggregation_k, "weight_per_pair": round(1.0 / aggregation_k, 8)})
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
            "server_mean_loss": round(sum(losses["segmentation"]) / len(losses["segmentation"]), 8) if losses["segmentation"] else "",
            "radar_distillation_mean_loss": round(sum(losses["radar_distillation"]) / len(losses["radar_distillation"]), 8) if losses["radar_distillation"] else "",
            "optical_distillation_mean_loss": round(sum(losses["optical_distillation"]) / len(losses["optical_distillation"]), 8) if losses["optical_distillation"] else "",
            "aggregation_performed": int(aggregation_performed), "aggregation_members": aggregation_members,
            "global_version": global_version, "modeled_transaction_s": round(estimate["duration_s"], 8),
        })
        # 星上本地训练只允许发生在不可见时段:把该对时钟直接推进到窗口结束
        pair.local_clock_s = max(pair.local_clock_s, contact["end_offset_s"])

    # 主循环结束后:各对在剩余时间内完成剩余的本地训练(不再有新事务与聚合)
    for pair in pairs.values():
        train_pair_offline(pair, horizon_s, config, radar_worker, optical_worker, radar_auxiliary, optical_auxiliary, radar_projection, optical_projection, attention_bias, criterion, device, epoch, local_log)
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
        "projection_distillation": "projR/projO MSE to detached cross_encoder features",
        "pixel_accuracy": pixel_accuracy, "mean_iou": mean_iou, "confusion_matrix": confusion,
        "last_ground_transaction_utc": utc_at(epoch, ground_available_s),
        "pending_matched_batches": {pair_id: len(pair.matched_batch_ids()) for pair_id, pair in pairs.items() if pair.matched_batch_ids()},
    }
    with (output_dir / "multimodal_training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    # 保存最终检查点:全局模型状态 + 服务器端跨模态编码器与地面头
    torch.save({"config": config, "global_version": global_version, "global_states": global_states, "cross_encoder_state": clone_state(cross_encoder), "ground_segmentation_head_state": clone_state(ground_head), "radar_projection_state": clone_state(radar_projection), "optical_projection_state": clone_state(optical_projection)}, output_dir / "multimodal_final_checkpoint.pt")
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
