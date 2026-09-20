import os
from typing import Optional, Callable, Tuple, List, Literal
import numpy as np
import torch
from torch.utils.data import Dataset
torch.set_printoptions(profile="full")
from PIL import Image
import rasterio
from rasterio.enums import Resampling
from rasterio.errors import NotGeoreferencedWarning
import warnings
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
import random
import pandas as pd
import re
from dataclasses import dataclass


def _normalize_tensor(img: torch.Tensor, method: str = 'minmax', mean=None, std=None) -> torch.Tensor:
    """Normalize a tensor image.

    Args:
        img: torch.Tensor, shape (C,H,W) or (H,W)
        method: 'minmax' or 'standard'
        mean, std: optional per-channel mean/std for 'standard'
    Returns:
        Normalized tensor (float)
    """
    if not torch.is_floating_point(img):
        img = img.float()

    if img.dim() == 3:
        C, H, W = img.shape
        flat = img.reshape(C, -1)
        if method == 'minmax':
            mins = flat.min(dim=1)[0].reshape(C, 1, 1)
            maxs = flat.max(dim=1)[0].reshape(C, 1, 1)
            denom = maxs - mins
            denom[denom == 0] = 1.0
            return (img - mins) / denom
        elif method == 'standard':
            if mean is None or std is None:
                mean_tensor = flat.mean(dim=1).reshape(C, 1, 1)
                std_tensor = flat.std(dim=1).reshape(C, 1, 1)
            else:
                mean_tensor = torch.tensor(mean, dtype=img.dtype, device=img.device).reshape(C, 1, 1)
                std_tensor = torch.tensor(std, dtype=img.dtype, device=img.device).reshape(C, 1, 1)
            std_tensor[std_tensor == 0] = 1.0
            return (img - mean_tensor) / std_tensor
    elif img.dim() == 2:
        if method == 'minmax':
            minv = img.min()
            maxv = img.max()
            denom = (maxv - minv) if (maxv - minv) != 0 else 1.0
            return (img - minv) / denom
        elif method == 'standard':
            meanv = img.mean() if mean is None else mean
            stdv = img.std() if std is None else std
            if stdv == 0:
                stdv = 1.0
            return (img - meanv) / stdv

    return img


def _update_channel_stats(
    sum_c: Optional[np.ndarray],
    sumsq_c: Optional[np.ndarray],
    count: int,
    arr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Accumulate per-channel sum/squared-sum/pixel-count for global stats."""
    arr64 = arr.astype(np.float64, copy=False)
    if arr64.ndim == 2:
        arr64 = arr64[np.newaxis, ...]

    c = arr64.shape[0]
    flat = arr64.reshape(c, -1)

    if sum_c is None:
        sum_c = np.zeros(c, dtype=np.float64)
        sumsq_c = np.zeros(c, dtype=np.float64)

    sum_c += flat.sum(axis=1)
    sumsq_c += np.square(flat).sum(axis=1)
    count += flat.shape[1]
    return sum_c, sumsq_c, count


def _finalize_channel_stats(sum_c: np.ndarray, sumsq_c: np.ndarray, count: int) -> Tuple[List[float], List[float]]:
    """Convert accumulated stats into mean/std lists."""
    if count <= 0:
        raise ValueError("Cannot compute normalization stats from empty data.")

    mean = sum_c / count
    var = (sumsq_c / count) - np.square(mean)
    var = np.maximum(var, 1e-12)
    std = np.sqrt(var)
    return mean.tolist(), std.tolist()


def _save_stats_to_csv(save_dir: str, filename: str, mean: List[float], std: List[float]) -> str:
    """Save per-channel mean/variance/std to CSV and return file path."""
    os.makedirs(save_dir, exist_ok=True)
    var = [float(s) ** 2 for s in std]
    rows = []
    for i, (m, v, s) in enumerate(zip(mean, var, std)):
        rows.append(
            {
                "channel": i,
                "mean": float(m),
                "variance": float(v),
                "std": float(s),
            }
        )

    out_path = os.path.join(save_dir, filename)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


def _load_stats_from_csv(save_dir: str, filename: str) -> Optional[Tuple[List[float], List[float]]]:
    """Load per-channel mean/std from CSV if file exists and has required columns."""
    path = os.path.join(save_dir, filename)
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)
    if "mean" not in df.columns:
        return None

    if "std" in df.columns:
        std = df["std"].astype(float).tolist()
    elif "variance" in df.columns:
        var = df["variance"].astype(float).tolist()
        std = [float(np.sqrt(max(v, 1e-12))) for v in var]
    else:
        return None

    mean = df["mean"].astype(float).tolist()
    if len(mean) == 0 or len(mean) != len(std):
        return None

    return mean, std


#whu-opt-sar dataset
class WHUOptSarPatchDataset(Dataset):
    """
    光学-SAR多模态遥感图像分割数据集，使用滑动窗口裁切扩充样本
    
    Args:
        root_dir: 数据集根目录
        split: 数据集划分，'train' 或 'val'
        train_ratio: 训练集比例，默认0.85
        patch_size: 裁切窗口大小，默认256
        stride_ratio: 步长相对于窗口大小的比例，默认0.9（即重叠10%）
        optical_dir: 光学图像子目录名
        sar_dir: SAR图像子目录名
        label_dir: 标签子目录名
        transform: 可选的数据增强变换
        random_seed: 随机种子，用于数据集划分
    """
    
    def __init__(
        self,
        root_dir: str,
        split: Literal['train', 'val'] = 'train',
        train_ratio: float = 0.8,
        patch_size: int = 256,
        stride_ratio: float = 0.9,
        optical_dir: str = 'optical',
        sar_dir: str = 'sar',
        label_dir: str = 'lbl',
        transform: Optional[Callable] = None,
        random_seed: int = 42,
        num_ratio: float = 1.0,
        normalize: bool = True,
        norm_type: str = 'standard',
        norm_mean: Optional[List[float]] = None,
        norm_std: Optional[List[float]] = None,
    ):
        self.root_dir = root_dir
        self.optical_dir = os.path.join(root_dir, optical_dir)
        self.sar_dir = os.path.join(root_dir, sar_dir)
        self.label_dir = os.path.join(root_dir, label_dir)
        self.split = split
        self.patch_size = patch_size
        self.stride = int(patch_size * stride_ratio)
        self.transform = transform
        self.num_ratio = num_ratio #用一部分数据集
        self.normalize = normalize
        self.norm_type = norm_type
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.stats_save_dir = os.path.join('.', 'Statistical_data', 'whu')
        self.optical_mean = None
        self.optical_std = None
        self.sar_mean = None
        self.sar_std = None

        
        # 原数据集的标签是0,10,20,...,70，将其映射到0,1,2,...,7
        self.label_mapping = {0: 0, 10: 1, 20: 2, 30: 3, 40: 4, 50: 5, 60: 6, 70: 7}
        
        # 获取所有文件
        all_files = self._get_file_list()
        
        # 划分训练集和验证集
        random.seed(random_seed)
        shuffled_files = all_files.copy()
        random.shuffle(shuffled_files)
        
        train_size = int(len(shuffled_files) * train_ratio)
        self.train_image_files = shuffled_files[:train_size]
        if split == 'train':
            self.image_files = self.train_image_files
        else:
            self.image_files = shuffled_files[train_size:]
        
        # 生成所有patch的索引
        self.patches = self._generate_patches()

        if self.normalize and self.norm_type == 'standard':
            if self.norm_mean is not None and self.norm_std is not None:
                # 兼容旧参数：若手动提供同一组均值/方差，则同时用于 optical/sar。
                self.optical_mean = self.norm_mean
                self.optical_std = self.norm_std
                self.sar_mean = self.norm_mean
                self.sar_std = self.norm_std
            else:
                optical_file = "optical_stats_train.csv"
                sar_file = "sar_stats_train.csv"

                optical_stats = _load_stats_from_csv(self.stats_save_dir, optical_file)
                sar_stats = _load_stats_from_csv(self.stats_save_dir, sar_file)

                if optical_stats is not None and sar_stats is not None:
                    self.optical_mean, self.optical_std = optical_stats
                    self.sar_mean, self.sar_std = sar_stats
                else:
                    self.optical_mean, self.optical_std = self._compute_global_stats(self.optical_dir, self.train_image_files)
                    self.sar_mean, self.sar_std = self._compute_global_stats(self.sar_dir, self.train_image_files)
                    _save_stats_to_csv(
                        self.stats_save_dir,
                        optical_file,
                        self.optical_mean,
                        self.optical_std,
                    )
                    _save_stats_to_csv(
                        self.stats_save_dir,
                        sar_file,
                        self.sar_mean,
                        self.sar_std,
                    )
        
    def _get_file_list(self) -> List[str]:
        """获取所有图像文件名"""
        optical_files = set([f for f in os.listdir(self.optical_dir) if f.endswith('.tif')])
        sar_files = set([f for f in os.listdir(self.sar_dir) if f.endswith('.tif')])
        label_files = set([f for f in os.listdir(self.label_dir) if f.endswith('.tif')])
        common_files = optical_files & sar_files & label_files
        return sorted(list(common_files))
    
    def _generate_patches(self) -> List[Tuple[int, int, int]]:
        """
        生成所有patch的索引
        返回列表，每个元素为 (file_idx, row_start, col_start)
        """
        patches = []
        
        for file_idx, filename in enumerate(self.image_files):
            # 读取一个标签文件获取图像尺寸
            label_path = os.path.join(self.label_dir, filename)
            with rasterio.open(label_path) as src:
                height, width = src.height, src.width
            
            # 计算滑动窗口的起始位置
            row_starts = list(range(0, height - self.patch_size + 1, self.stride))
            col_starts = list(range(0, width - self.patch_size + 1, self.stride))
            
            # 确保覆盖到边界
            if row_starts[-1] + self.patch_size < height:
                row_starts.append(height - self.patch_size)
            if col_starts[-1] + self.patch_size < width:
                col_starts.append(width - self.patch_size)
            
            # 生成所有窗口位置
            for row in row_starts:
                for col in col_starts:
                    patches.append((file_idx, row, col))
            
        # 随机打乱
        shuffled_patches = patches.copy()
        random.shuffle(shuffled_patches)

        n = len(shuffled_patches)
        target = int(n * self.num_ratio)

        if target <= 0:
            return []

        # 当 num_ratio <= 1 时，直接截取子集（降采样或等于原始数量）
        if self.num_ratio <= 1.0:
            return shuffled_patches[:target]

        # 当 num_ratio > 1 时，需要扩增到目标数量。
        # 如果目标不超过原始数量，直接截取；否则使用有放回采样补足数量。
        if target <= n:
            return shuffled_patches[:target]

        # target > n: 使用有放回采样扩增到目标数量
        # 直接从打乱后的补丁列表中有放回采样 target 个
        expanded = random.choices(shuffled_patches, k=target)
        return expanded
    
    def __len__(self) -> int:
        return len(self.patches)
    
    def _read_tif(self, file_path: str) -> np.ndarray:
        """使用rasterio读取tif文件"""
        with rasterio.open(file_path) as src:
            image = src.read()
        return image
    
    def _read_patch(self, file_path: str, row: int, col: int) -> np.ndarray:
        """读取指定位置的patch"""
        with rasterio.open(file_path) as src:
            # 读取窗口：(col_off, row_off, width, height)
            window = ((row, row + self.patch_size), (col, col + self.patch_size))
            patch = src.read(window=window)
        return patch

    def _compute_global_stats(self, folder_path: str, file_list: Optional[List[str]] = None) -> Tuple[List[float], List[float]]:
        """Compute per-channel mean/std over all images in current split."""
        if file_list is None:
            file_list = self.image_files
        sum_c, sumsq_c, count = None, None, 0
        for filename in file_list:
            file_path = os.path.join(folder_path, filename)
            arr = self._read_tif(file_path)
            sum_c, sumsq_c, count = _update_channel_stats(sum_c, sumsq_c, count, arr)
        return _finalize_channel_stats(sum_c, sumsq_c, count)
    
    def _map_labels(self, label: np.ndarray) -> np.ndarray:
        """
        映射标签值：0,10,20,...,70 -> 0,1,2,...,7
        
        Args:
            label: 原始标签数组
            
        Returns:
            映射后的标签数组
        """
        mapped_label = np.zeros_like(label, dtype=np.int64)
        for old_val, new_val in self.label_mapping.items():
            mapped_label[label == old_val] = new_val
        return mapped_label
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取一个patch样本
        
        Returns:
            (optical_patch, sar_patch, label_patch):
            - optical_patch: 光学图像patch，shape (C, patch_size, patch_size)
            - sar_patch: SAR图像patch，shape (C, patch_size, patch_size)
            - label_patch: 标签patch，shape (patch_size, patch_size)
        """
        file_idx, row, col = self.patches[idx]
        filename = self.image_files[file_idx]
        
        # 构建文件路径
        optical_path = os.path.join(self.optical_dir, filename)
        sar_path = os.path.join(self.sar_dir, filename)
        label_path = os.path.join(self.label_dir, filename)
        
        # 读取patch
        optical_patch = self._read_patch(optical_path, row, col)
        sar_patch = self._read_patch(sar_path, row, col)
        label_patch = self._read_patch(label_path, row, col)
        
        # 如果标签只有一个通道，去掉通道维度
        if label_patch.shape[0] == 1:
            label_patch = label_patch[0]
        
        # 映射标签值：0,10,20,...,70 -> 0,1,2,...,7
        label_patch = self._map_labels(label_patch)
        
        # 转换为torch张量
        optical_patch = torch.from_numpy(optical_patch).float()
        sar_patch = torch.from_numpy(sar_patch).float()
        label_patch = torch.from_numpy(label_patch).long() if label_patch.ndim == 2 else torch.from_numpy(label_patch).float()
        
        # 应用数据增强
        # 归一化（在数据增强前或后都可以，根据需要这里放在增强之前）
        if self.normalize:
            optical_patch = _normalize_tensor(
                optical_patch,
                method=self.norm_type,
                mean=self.optical_mean if self.norm_type == 'standard' else self.norm_mean,
                std=self.optical_std if self.norm_type == 'standard' else self.norm_std,
            )
            sar_patch = _normalize_tensor(
                sar_patch,
                method=self.norm_type,
                mean=self.sar_mean if self.norm_type == 'standard' else self.norm_mean,
                std=self.sar_std if self.norm_type == 'standard' else self.norm_std,
            )

        if self.transform is not None:
            optical_patch, sar_patch, label_patch = self.transform(optical_patch, sar_patch, label_patch)
        
        return optical_patch, sar_patch, label_patch


#Houston2013
@dataclass
class PatchIndex:
    top: int
    left: int

def _read_tif(path: str) -> np.ndarray:
    with rasterio.open(path) as ds:
        arr = ds.read()
    return arr

def _ensure_2d_label(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3 and arr.shape[0] == 1:
        return arr[0]
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Label array must be (H, W) or (1, H, W), got {arr.shape}.")

CASI_FILE = "2013_IEEE_GRSS_DF_Contest_CASI.tif"
LIDAR_FILE = "2013_IEEE_GRSS_DF_Contest_LiDAR.tif"
TR_LABEL_FILE = "2013_IEEE_GRSS_DF_Contest_Samples_TR.tif"
VA_LABEL_FILE = "2013_IEEE_GRSS_DF_Contest_Samples_VA.tif"

class Houston2013PatchDataset(Dataset):
    """Houston2013 HSI + LiDAR patch dataset for sparse labels.

    Returns:
        hsi: torch.FloatTensor, shape (144, patch_size, patch_size)
        lidar: torch.FloatTensor, shape (1, patch_size, patch_size)
        label: torch.LongTensor, shape (patch_size, patch_size), value range [-1, 14]
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        patch_size: int = 256,
        stride: int = 256,
        ignore_index: int = 0,
        drop_empty: bool = True,
        transform: Optional[Callable] = None,
        return_coords: bool = False,
        normalize: bool = True,
        norm_type: str = "standard",
    ) -> None:
        super().__init__()

        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'.")

        self.root_dir = root_dir
        self.split = split
        self.patch_size = patch_size
        self.stride = stride
        self.ignore_index = ignore_index
        self.drop_empty = drop_empty
        self.transform = transform
        self.return_coords = return_coords
        self.normalize = normalize
        self.norm_type = norm_type
        self.stats_save_dir = os.path.join('.', 'Statistical_data', 'Houston2013')
        self.hsi_mean = None
        self.hsi_std = None
        self.lidar_mean = None
        self.lidar_std = None

        self.hsi = _read_tif(os.path.join(root_dir, CASI_FILE)).astype(np.float32)
        self.lidar = _read_tif(os.path.join(root_dir, LIDAR_FILE)).astype(np.float32)

        label_file = TR_LABEL_FILE if split == "train" else VA_LABEL_FILE
        self.label = _ensure_2d_label(_read_tif(os.path.join(root_dir, label_file)))

        if self.hsi.shape[0] != 144:
            raise ValueError(f"HSI channels should be 144, got {self.hsi.shape[0]}.")
        if self.lidar.ndim != 3 or self.lidar.shape[0] != 1:
            raise ValueError(f"LiDAR should be (1, H, W), got {self.lidar.shape}.")

        h, w = self.label.shape
        if self.hsi.shape[1:] != (h, w) or self.lidar.shape[1:] != (h, w):
            raise ValueError(
                "HSI, LiDAR, and label spatial sizes are inconsistent: "
                f"HSI={self.hsi.shape}, LiDAR={self.lidar.shape}, label={self.label.shape}."
            )

        hsi_file = "hsi_stats.csv"
        lidar_file = "lidar_stats.csv"

        hsi_stats = _load_stats_from_csv(self.stats_save_dir, hsi_file)
        lidar_stats = _load_stats_from_csv(self.stats_save_dir, lidar_file)

        if hsi_stats is not None:
            self.hsi_mean, self.hsi_std = hsi_stats
        else:
            self.hsi_mean, self.hsi_std = self._compute_global_stats(self.hsi)
            _save_stats_to_csv(
                self.stats_save_dir,
                hsi_file,
                self.hsi_mean,
                self.hsi_std,
            )

        if lidar_stats is not None:
            self.lidar_mean, self.lidar_std = lidar_stats
        else:
            self.lidar_mean, self.lidar_std = self._compute_global_stats(self.lidar)
            _save_stats_to_csv(
                self.stats_save_dir,
                lidar_file,
                self.lidar_mean,
                self.lidar_std,
            )

        self.patch_indices = self._build_patch_indices(h, w)

    def _compute_global_stats(self, arr: np.ndarray) -> Tuple[List[float], List[float]]:
        """Compute per-channel mean/std from a (C,H,W) array."""
        sum_c, sumsq_c, count = _update_channel_stats(None, None, 0, arr)
        return _finalize_channel_stats(sum_c, sumsq_c, count)

    def _build_patch_indices(self, h: int, w: int) -> List[PatchIndex]:
        ps = self.patch_size
        st = self.stride

        def build_starts(length: int) -> List[int]:
            if length <= ps:
                return [0]
            starts = list(range(0, length - ps + 1, st))
            last = length - ps
            if starts[-1] != last:
                starts.append(last)
            return starts

        top_starts = build_starts(h)
        left_starts = build_starts(w)

        indices: List[PatchIndex] = []
        for top in top_starts:
            for left in left_starts:
                patch_label = self.label[top : top + ps, left : left + ps]
                if self.drop_empty and not np.any(patch_label != self.ignore_index):
                    continue
                indices.append(PatchIndex(top=top, left=left))

        return indices

    def __len__(self) -> int:
        return len(self.patch_indices)

    def __getitem__(self, idx: int):
        p = self.patch_indices[idx]
        top, left = p.top, p.left
        ps = self.patch_size

        hsi_patch = self.hsi[:, top : top + ps, left : left + ps]
        lidar_patch = self.lidar[:, top : top + ps, left : left + ps]
        label_np = self.label[top : top + ps, left : left + ps]

        hsi = torch.from_numpy(hsi_patch).float()
        lidar = torch.from_numpy(lidar_patch).float()
        # Houston 原始标签为 0..15，其中 0 为未标注；整体减 1 后变为 -1..14。
        label = torch.from_numpy(label_np).long() - 1

        if self.normalize:
            hsi = _normalize_tensor(
                hsi,
                method=self.norm_type,
                mean=self.hsi_mean if self.norm_type == "standard" else None,
                std=self.hsi_std if self.norm_type == "standard" else None,
            )
            lidar = _normalize_tensor(
                lidar,
                method=self.norm_type,
                mean=self.lidar_mean if self.norm_type == "standard" else None,
                std=self.lidar_std if self.norm_type == "standard" else None,
            )

        if self.transform is not None:
            transformed = self.transform(hsi, lidar, label)
            if not isinstance(transformed, (tuple, list)) or len(transformed) != 3:
                raise ValueError("transform must return (hsi, lidar, label).")
            hsi, lidar, label = transformed

        if self.return_coords:
            return hsi, lidar, label, {"top": top, "left": left}
        return hsi, lidar, label