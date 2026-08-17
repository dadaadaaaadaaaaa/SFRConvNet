# analysis_utils.py
import os
import sys
import gc
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import h5py


# ==========================================
# 1. Project Setup & Imports
# ==========================================
def setup_project_env():
    """Found project root and set up environment."""
    current_path = Path.cwd()
    project_root = None
    markers = ["Experiment", "raw_data_processing", ".config"]

    # Try to find root from CWD (Notebook execution context)
    cand_path = current_path
    while cand_path != cand_path.parent:
        if any((cand_path / m).exists() for m in markers):
            project_root = cand_path
            break
        cand_path = cand_path.parent

    # Fallback: relative to this file if it's imported
    if project_root is None:
        try:
            # If __file__ is available
            project_root = Path(__file__).resolve().parent.parent.parent
        except NameError:
            # Fallback for some interactive environments
            project_root = Path(os.path.abspath(os.path.join(os.getcwd(), "..", "..")))

    # Add to sys.path
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Ensure CWD is project root (fixes .config discovery in prepare_data.py)
    try:
        if Path.cwd() != project_root:
            os.chdir(project_root)
            print(f"[INFO] Changed CWD to Project Root: {project_root}")
    except Exception as e:
        print(f"[WARN] Could not switch CWD: {e}")

    return project_root


# Initialize root immediately for imports in this file
PROJECT_ROOT = setup_project_env()

# Project Imports
try:
    from raw_data_processing.galaxy_data import GalDataSet
    from Experiment.target_transform import inverse_transform_sfr, transform_sfr
    from Experiment.models import (
        MACNetRes_mbh,
        AccretionTransformer,
        MACNetFiLM,
        FourierPositionalEncoding2D,
        FiLMConditioner,
        MLP,
        SFRConvNet,
    )

    # Import Preprocess paths if available
    try:
        from Experiment.prepare_data import Exp_eg_folder_paths, Exp_dg_folder_paths
    except ImportError:
        Exp_eg_folder_paths = []
        Exp_dg_folder_paths = []
except ImportError as e:
    print(f"[WARN] Project imports failed in analysis_utils: {e}")


# ==========================================
# 2. Legacy Classes (from Exp2)
# ==========================================
class LegacyTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, p_drop: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=p_drop, batch_first=True)
        self.drop1 = nn.Dropout(p_drop)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff, p_drop)

    def forward(self, x, gamma=None, beta=None):
        if gamma is not None and beta is not None:
            x = gamma * x + beta
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop1(h)
        if gamma is not None and beta is not None:
            x = gamma * x + beta
        h = self.norm2(x)
        h = self.mlp(h)
        x = x + h
        return x


class LegacyAccretionTransformer(nn.Module):
    def __init__(
        self, c_in, d_model=256, n_layers=8, n_heads=8, d_ff=1024, pos_num_bands=32, pos_max_freq=64.0, p_drop=0.1
    ):
        super().__init__()
        self.posenc = FourierPositionalEncoding2D(
            num_bands=pos_num_bands, max_frequency=pos_max_freq, include_input=True
        )
        d_pos = self.posenc.out_dim
        self.content_proj = nn.Conv2d(c_in, d_model // 2, kernel_size=1)
        self.pos_proj = nn.Conv2d(d_pos, d_model // 2, kernel_size=1)
        self.drop = nn.Dropout(p_drop)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.blocks = nn.ModuleList([LegacyTransformerBlock(d_model, n_heads, d_ff, p_drop) for _ in range(n_layers)])
        self.film = FiLMConditioner(n_layers, d_model)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x, r, theta, bh_mass):
        B, C, H, W = x.shape
        pos = self.posenc(r, theta, normalize=True)
        feat = self.content_proj(x) + self.pos_proj(pos)
        feat = torch.cat([feat, self.drop(feat)], dim=1)
        tokens = feat.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        gammas, betas = self.film(bh_mass)
        x_tok = tokens
        for i, blk in enumerate(self.blocks):
            gamma = gammas[i].expand(B, x_tok.size(1), -1)
            beta = betas[i].expand(B, x_tok.size(1), -1)
            x_tok = blk(x_tok, gamma=gamma, beta=beta)
        return self.head(x_tok[:, 0, :]).squeeze(-1)


# ==========================================
# 3. Helper Functions
# ==========================================
def ensure_tensor(arr, dtype=torch.float32):
    if arr is None:
        return None
    if isinstance(arr, torch.Tensor):
        return arr.to(dtype=dtype).contiguous()
    return torch.as_tensor(arr, dtype=dtype).contiguous()


def ensure_numpy(arr):
    if arr is None:
        return None
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def get_baseline_column(df):
    if "y_baseline" in df.columns:
        return "y_baseline"
    return None


def get_baseline_error_column(df):
    if "baseline_error" in df.columns:
        return "baseline_error"
    return None


def _target_metadata_from(obj=None):
    """Return normalized target metadata from a dataset or prediction DataFrame."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        metadata = obj
    elif isinstance(obj, pd.DataFrame):
        metadata = obj.attrs.get("target_metadata", {})
    else:
        metadata = getattr(obj, "target_metadata", {})
    metadata = dict(metadata or {})
    normalized = {}
    for key, value in metadata.items():
        normalized[key.removeprefix("target_")] = value
    return normalized


def _physical_sfr(values, metadata):
    """Invert the configured training transform; legacy runs use log10(SFR+floor)."""
    metadata = dict(metadata or {})
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    physical = np.full(values.shape, np.nan, dtype=float)
    physical[finite] = inverse_transform_sfr(
        values[finite],
        transform=metadata.get("transform", "log10_floor"),
        scale=float(metadata.get("scale", 1.0)),
        floor=float(metadata.get("floor", 1e-9)),
    )
    return physical


def get_target_label(df=None):
    metadata = _target_metadata_from(df)
    transform = metadata.get("transform")
    if transform == "log10_1p":
        scale = float(metadata.get("scale", 1.0))
        return rf"$\log_{{10}}[1+\mathrm{{SFR}}/({scale:g}\,M_\odot\,\mathrm{{yr}}^{{-1}})]$"
    return r"$\log_{10}(\mathrm{SFR}/[M_\odot\,\mathrm{yr}^{-1}])$"


def get_baseline_label(df=None):
    return "Baseline (KS)"


def find_latest_model(results_root: Path, exp_prefix: str) -> Path:
    if not results_root.exists():
        return None
    candidates = list(results_root.glob(f"{exp_prefix}*/*.pth"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


# ==========================================
# 4. Datasets
# ==========================================
def _cache_prefix_from_split_path(path):
    path = Path(path)
    name = path.name
    for suffix in ("train.pt", "val.pt", "test.pt", "out.pt"):
        if name.endswith(suffix):
            return path.with_name(name[: -len(suffix)])
    return None


def load_cache_metadata(path):
    """Load channel metadata associated with a cache split or cache prefix."""
    path = Path(path)
    prefix = _cache_prefix_from_split_path(path)
    if prefix is None:
        prefix = path
    metadata_path = Path(f"{prefix}metadata.json")
    if not metadata_path.exists():
        return {}
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_norm_stats(norm_path):
    """Load normalization arrays plus optional channel metadata."""
    with np.load(norm_path, allow_pickle=False) as data:
        result = {
            "mean": data["mean"],
            "std": data["std"],
            "input_channels": tuple(data["input_channels"].tolist()) if "input_channels" in data else (),
            "excluded_channels": tuple(data["excluded_channels"].tolist()) if "excluded_channels" in data else (),
        }
        scalar_keys = (
            "target_quantity", "target_name", "target_dataset", "target_reduction",
            "target_units", "target_transform", "target_scale", "target_floor",
            "includes_zero_targets",
        )
        for key in scalar_keys:
            if key in data:
                result[key] = data[key].item()
        return result


class CachedPTDataset:
    def __init__(self, pt_path: Path):
        self.path = pt_path
        if not pt_path.exists():
            raise FileNotFoundError(f"Cached dataset not found: {pt_path}")

        loaded = torch.load(pt_path, map_location="cpu", weights_only=False)
        items = list(loaded)

        self.coord = None
        self.y_baseline = None
        self.baseline = None
        self.groups = None
        self.n_groups = None
        self.time = None
        self.type = "cached"
        channel_metadata = load_cache_metadata(pt_path)
        self.input_channels = tuple(channel_metadata.get("input_channels", ()))
        self.excluded_channels = tuple(channel_metadata.get("excluded_channels", ()))
        self.target_metadata = {
            key.removeprefix("target_"): value
            for key, value in channel_metadata.items()
            if key.startswith("target_")
        }
        if "includes_zero_targets" in channel_metadata:
            self.target_metadata["includes_zero_targets"] = bool(channel_metadata["includes_zero_targets"])

        # Current SFR cache format from prepare_data.py:
        # (x, coord, target_sfr, transformed_ks_sfr, groups, n_groups, time)
        if len(items) >= 7 and isinstance(items[1], (torch.Tensor, np.ndarray)) and np.asarray(items[1]).ndim >= 2:
            self.x, self.coord, self.y, self.y_baseline, self.groups, self.n_groups, self.time = items[:7]
            self.kind = "vit"
        elif len(items) >= 5 and isinstance(items[1], (torch.Tensor, np.ndarray)) and np.asarray(items[1]).ndim >= 2:
            self.x, self.coord, self.y, self.y_baseline, self.groups = items[:5]
            self.kind = "vit"
        elif len(items) >= 5:
            self.x, self.y, self.y_baseline, self.groups, self.n_groups = items[:5]
            self.kind = "macnet"
        else:
            self.x, self.y, self.y_baseline = items[:3]
            self.kind = "macnet"

        self.x = ensure_tensor(self.x)
        if self.input_channels and len(self.input_channels) != self.x.shape[1]:
            raise RuntimeError(
                f"Cache '{pt_path}' has {self.x.shape[1]} tensor channels but metadata lists "
                f"{len(self.input_channels)} channels: {self.input_channels}"
            )
        self.coord = ensure_tensor(self.coord)
        self.y = ensure_tensor(self.y).view(-1, 1)
        self.y_baseline = ensure_tensor(self.y_baseline).view(-1, 1) if self.y_baseline is not None else None
        self.baseline = self.y_baseline
        self.conditioning_scalar = self.y_baseline
        self.conditioning_variable = channel_metadata.get("conditioning_variable")
        if self.conditioning_variable is None:
            # Compatibility with instantaneous-SFR caches created before this
            # field was recorded explicitly.
            if channel_metadata.get("target_quantity") == "instantaneous_sfr":
                self.conditioning_variable = "sfr_ks"
            else:
                self.conditioning_variable = "unknown"
        if self.conditioning_variable == "sfr_ks":
            self.sfr_ks_transformed = self.conditioning_scalar
        self.time = ensure_tensor(self.time) if self.time is not None else None

        if self.groups is not None:
            self.group_names = None
            groups_arr = np.asarray(self.groups)
            if groups_arr.dtype.kind in ("S", "U"):
                # 如果 group 本身就是字符串 numpy 数组
                self.group_names = groups_arr
                _, indices = np.unique(groups_arr, return_inverse=True)
                self.groups = torch.as_tensor(indices, dtype=torch.long)
            elif isinstance(self.groups, (list, tuple)) and len(self.groups) > 0 and isinstance(self.groups[0], str):
                # 如果 group 是一组字符串列表
                self.group_names = np.array(self.groups)
                _, indices = np.unique(self.group_names, return_inverse=True)
                self.groups = torch.as_tensor(indices, dtype=torch.long)
            else:
                try:
                    self.groups = torch.as_tensor(self.groups)
                except Exception as e:
                    print(f"Warning: Could not automatically convert groups to tensor. {e}")

    def __len__(self):
        return len(self.y)


class HDF5Dataset:
    def __init__(
        self,
        hdf5_conf,
        resolution="coarse",
        mean=None,
        std=None,
        log_transform=True,
        mirror=True,
        balance=None,
        input_channels=None,
        exclude_channels=None,
        reference_dataset=None,
        target_metadata=None,
    ):
        if isinstance(hdf5_conf, (list, tuple)):
            self.folder_path = hdf5_conf[0]
        else:
            self.folder_path = hdf5_conf

        self.type = os.path.basename(os.path.dirname(self.folder_path)).lower()

        loaded = GalDataSet().load_data((self.folder_path, 0))
        loaded.drop_invalid(require_positive_targets=False, require_positive_ks=False)

        if reference_dataset is not None:
            if input_channels is None:
                input_channels = getattr(reference_dataset, "input_channels", None) or None
            if exclude_channels is None:
                exclude_channels = getattr(reference_dataset, "excluded_channels", None) or None
            if target_metadata is None:
                target_metadata = _target_metadata_from(reference_dataset)
        if input_channels is not None and exclude_channels is not None:
            # An explicit include list is sufficient and avoids ambiguous rules.
            exclude_channels = None
        loaded.select_input_channels(include=input_channels, exclude=exclude_channels)
        self.input_channels = loaded.input_channels
        self.excluded_channels = tuple(exclude_channels or ())

        if mirror:
            loaded.mirror_data()
        if balance is not None:
            loaded.balance_groups(method=balance)

        if mean is not None and std is not None:
            mean_shape = np.asarray(mean).shape
            norm_channels = int(mean_shape[0] if len(mean_shape) == 1 else mean_shape[1])
            if norm_channels != len(self.input_channels):
                raise ValueError(
                    f"Normalization has {norm_channels} channels but HDF5 input selection has "
                    f"{len(self.input_channels)} ({self.input_channels}). Pass input_channels, "
                    "exclude_channels, or reference_dataset from the matching cached experiment."
                )
            self.x = loaded.standardize(mean, std, log_transform=log_transform)
        else:
            self.x, _, _ = loaded.standardize(log_transform=log_transform)
        self.coord = loaded.coordinate
        raw_target_metadata = loaded.target_metadata
        if raw_target_metadata.get("quantity") != "instantaneous_sfr":
            raise ValueError(
                f"HDF5 target is {raw_target_metadata.get('quantity')!r}, expected 'instantaneous_sfr'. "
                "Regenerate the processed HDF5 snapshots."
            )
        self.target_metadata = dict(target_metadata or raw_target_metadata)
        self.target_metadata.setdefault("transform", "log10_1p")
        self.target_metadata.setdefault("scale", 1.0)
        self.target_metadata.setdefault("floor", 1e-9)
        for key in ("quantity", "dataset", "reduction", "units"):
            if key in raw_target_metadata and key in self.target_metadata:
                if raw_target_metadata[key] != self.target_metadata[key]:
                    raise ValueError(
                        f"Raw HDF5 target {key}={raw_target_metadata[key]!r} does not match "
                        f"reference target {self.target_metadata[key]!r}."
                    )
        transform_options = {
            "transform": self.target_metadata["transform"],
            "scale": float(self.target_metadata["scale"]),
            "floor": float(self.target_metadata["floor"]),
        }
        self.y = transform_sfr(loaded.sfr_target, **transform_options)
        # This is the transformed KS-law SFR conditioner used by SFRConvNet
        # when use_film=True; it is not black-hole mass.
        self.y_baseline = transform_sfr(loaded.sfr_ks, **transform_options)
        self.baseline = self.y_baseline
        self.conditioning_scalar = self.y_baseline
        self.conditioning_variable = "sfr_ks"
        self.sfr_ks_transformed = self.y_baseline
        self.time = loaded.time
        self.mdot_edd = np.array(loaded._raw_data["mdot_edd"]) if "mdot_edd" in loaded._raw_data else None
        self.sfr_target = loaded.sfr_target

        self.kind = "vit" if self.coord is not None else "macnet"

    def __len__(self):
        return len(self.y)


# ==========================================
# 5. Model Loading
# ==========================================
def load_analysis_model(model_path, device, datasets_dict=None):
    """Refactored model loading logic."""
    model_path = Path(model_path)
    state = torch.load(model_path, map_location=device, weights_only=False)
    raw_state = state.get("model", state.get("state_dict", state))
    keys = list(raw_state.keys())

    # Check for hyperparams FIRST
    config_path = model_path.parent / "hyperparameters.json"
    loaded_params = {}
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                loaded_params = json.load(f)
        except:
            pass
    conditioning_variable = (
        state.get("conditioning_variable") if isinstance(state, dict) else None
    ) or loaded_params.get("conditioning_variable")
    if conditioning_variable is None and loaded_params.get("target_quantity") == "instantaneous_sfr":
        # Compatibility with SFR checkpoints created before this metadata was
        # saved explicitly. Their PT fourth field is transformed KS-law SFR.
        conditioning_variable = "sfr_ks"

    # Infer Type
    model_type = "macnet"
    if loaded_params and "model_type" in loaded_params:
        model_type = loaded_params["model_type"]
    else:
        if any(k.startswith("film_generator") for k in keys):
            model_type = "resfilm"
        elif any(k.startswith("cls_token") or k.startswith("posenc") for k in keys) or any(
            k.startswith("blocks") for k in keys
        ):
            if any(k.startswith("stem.") for k in keys):
                model_type = "convnet"
            else:
                model_type = "vit"

    # Infer Input Channels. New checkpoints/cache metadata are authoritative;
    # legacy runs still fall back to the supplied analysis dataset.
    checkpoint_channels = tuple(state.get("input_channels", ())) if isinstance(state, dict) else ()
    checkpoint_n_channels = state.get("n_input_channels") if isinstance(state, dict) else None
    configured_n_channels = loaded_params.get("n_input_channels") if loaded_params else None
    in_ch = int(checkpoint_n_channels or configured_n_channels or 14)
    dataset_channels = ()
    checkpoint_target_metadata = _target_metadata_from(
        state.get("target_metadata", {}) if isinstance(state, dict) else {}
    )
    if not checkpoint_target_metadata and loaded_params:
        checkpoint_target_metadata = _target_metadata_from(loaded_params.get("target_metadata", {}))
    if datasets_dict:
        ref_ds = next((ds for ds in datasets_dict.values() if ds is not None), None)
        if ref_ds:
            dataset_n_channels = int(ref_ds.x.shape[1])
            if (checkpoint_n_channels or configured_n_channels) and dataset_n_channels != in_ch:
                raise ValueError(
                    f"Model expects {in_ch} input channels but analysis dataset has "
                    f"{dataset_n_channels}."
                )
            dataset_channels = tuple(getattr(ref_ds, "input_channels", ()))
            if checkpoint_channels and dataset_channels and checkpoint_channels != dataset_channels:
                raise ValueError(
                    f"Model channels {checkpoint_channels} do not match dataset channels "
                    f"{dataset_channels}."
                )
            in_ch = dataset_n_channels
            dataset_target_metadata = _target_metadata_from(ref_ds)
            if bool(checkpoint_target_metadata) != bool(dataset_target_metadata):
                raise ValueError(
                    "Model and analysis dataset do not both provide target metadata. "
                    "Do not mix a legacy cumulative-SFR run with an instantaneous-SFR cache."
                )
            for key in ("quantity", "dataset", "reduction", "transform", "scale", "floor"):
                if key in checkpoint_target_metadata or key in dataset_target_metadata:
                    if checkpoint_target_metadata.get(key) != dataset_target_metadata.get(key):
                        raise ValueError(
                            f"Model target {key}={checkpoint_target_metadata.get(key)!r} does not match "
                            f"dataset target {dataset_target_metadata.get(key)!r}."
                        )

    # Initialize
    model = None
    if model_type == "vit":
        # Default Params
        d_model = 256
        n_layers = 8
        n_heads = 4
        d_ff = 1024
        pos_num_bands = 32

        # Override from JSON
        if loaded_params:
            d_model = loaded_params.get("d_model", d_model)
            n_layers = loaded_params.get("n_layers", n_layers)
            n_heads = loaded_params.get("n_heads", n_heads)
            d_ff = loaded_params.get("d_ff", d_ff)
            pos_num_bands = loaded_params.get("pos_num_bands", pos_num_bands)
        else:
            # Fallback Inference
            if "cls_token" in raw_state:
                d_model = raw_state["cls_token"].shape[-1]
            block_ids = [int(k.split(".")[1]) for k in keys if k.startswith("blocks.")]
            n_layers = max(block_ids) + 1 if block_ids else 8
            for k in keys:
                if "mlp.fc1.weight" in k:
                    d_ff = raw_state[k].shape[0]
                    break
            n_heads = max(1, d_model // 32)
            if "posenc.freq_bands" in raw_state:
                pos_num_bands = raw_state["posenc.freq_bands"].shape[0]

        # Check Legacy
        is_legacy = False
        if "content_proj.weight" in raw_state:
            cp_shape = raw_state["content_proj.weight"].shape
            if cp_shape[0] == d_model // 2:
                is_legacy = True

        if is_legacy:
            model = LegacyAccretionTransformer(
                c_in=in_ch, d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_ff=d_ff, pos_num_bands=pos_num_bands
            ).to(device)
        else:
            model = AccretionTransformer(
                c_in=in_ch, d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_ff=d_ff, pos_num_bands=pos_num_bands
            ).to(device)

    elif model_type == "convnet":
        base_ch = 64
        stage_depths = (2, 2, 2, 2)
        pos_num_bands = 32
        pos_max_freq = 64.0
        p_drop = 0.1
        se_reduction = 4
        use_film = False

        if loaded_params:
            base_ch = loaded_params.get("base_ch", base_ch)
            sd_val = loaded_params.get("stage_depths", "2,2,2,2")
            if isinstance(sd_val, str):
                stage_depths = tuple(int(x) for x in sd_val.split(","))
            elif isinstance(sd_val, (list, tuple)):
                stage_depths = tuple(sd_val)
            pos_num_bands = loaded_params.get("pos_num_bands", pos_num_bands)
            pos_max_freq = loaded_params.get("pos_max_freq", pos_max_freq)
            p_drop = loaded_params.get("p_drop", p_drop)
            se_reduction = loaded_params.get("se_reduction", se_reduction)
            use_film = loaded_params.get("use_film", use_film)

        if use_film and conditioning_variable != "sfr_ks":
            raise RuntimeError(
                "SFRConvNet checkpoint enables FiLM but does not declare the expected "
                f"conditioning_variable='sfr_ks' (got {conditioning_variable!r})."
            )

        model = SFRConvNet(
            c_in=in_ch,
            base_ch=base_ch,
            stage_depths=stage_depths,
            pos_num_bands=pos_num_bands,
            pos_max_freq=pos_max_freq,
            p_drop=p_drop,
            se_reduction=se_reduction,
            use_film=use_film,
        ).to(device)

    elif model_type == "resfilm":
        if conditioning_variable == "sfr_ks":
            raise RuntimeError(
                "MACNetFiLM is a legacy BH-mass-conditioned model and cannot be "
                "loaded with an SFR cache whose conditioning variable is 'sfr_ks'."
            )
        model = MACNetFiLM(in_channels=in_ch).to(device)
    else:
        if conditioning_variable == "sfr_ks":
            raise RuntimeError(
                "MACNetRes_mbh is a legacy BH-mass-conditioned model and cannot be "
                "loaded with an SFR cache whose conditioning variable is 'sfr_ks'."
            )
        model = MACNetRes_mbh(in_channels=in_ch).to(device)

    try:
        model.load_state_dict(raw_state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Checkpoint '{model_path}' is not structurally compatible with the inferred {model_type} model."
        ) from exc

    model.eval()
    model.conditioning_variable = conditioning_variable or "unknown"
    model.input_channels = checkpoint_channels or dataset_channels
    model.target_metadata = checkpoint_target_metadata
    return model, model_type


# ==========================================
# 6. Prediction
# ==========================================
@torch.no_grad()
def predict_dataset(ds, model, device, model_type, batch_size=32, return_feats=False):
    if ds is None:
        return (None, None, None) if return_feats else None

    x_all = ensure_tensor(ds.x)
    if x_all is None or len(x_all) == 0:
        empty_df = pd.DataFrame(columns=["y_true", "y_pred", "error", "type"])
        return (empty_df, None, None) if return_feats else empty_df

    x_data = ensure_numpy(x_all)
    conditioning_all = ensure_tensor(
        getattr(ds, "conditioning_scalar", getattr(ds, "y_baseline", None))
    )
    dataset_conditioning = getattr(ds, "conditioning_variable", "unknown")
    model_conditioning = getattr(model, "conditioning_variable", "unknown")
    if (
        dataset_conditioning != "unknown"
        and model_conditioning != "unknown"
        and dataset_conditioning != model_conditioning
    ):
        raise RuntimeError(
            "Model and dataset conditioning semantics differ: "
            f"model={model_conditioning!r}, dataset={dataset_conditioning!r}."
        )
    coord_all = ensure_tensor(ds.coord) if getattr(ds, "coord", None) is not None else None

    total = len(ds)
    preds = []
    feats_list = []

    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    # ViT remains conservative because attention memory grows quickly. ConvNet
    # inference uses the caller-provided value; gradients are disabled here, so
    # it can normally use at least the training batch size.
    bs = 8 if model_type == "vit" else batch_size

    # Hook for features if requested
    hook_handle = None
    if return_feats:
        if model_type == "vit":
            def get_activation(storage):
                def hook(model, input, output):
                    storage.append(output[:, 0, :].detach().cpu())
                return hook

            if hasattr(model, "blocks") and len(model.blocks) > 0:
                hook_handle = model.blocks[-1].register_forward_hook(get_activation(feats_list))
        elif model_type == "convnet":
            def get_activation_convnet(storage):
                def hook(model, input, output):
                    storage.append(output.reshape(output.shape[0], -1).detach().cpu())
                return hook

            if hasattr(model, "head") and len(model.head) >= 3:
                hook_handle = model.head[2].register_forward_hook(get_activation_convnet(feats_list))

    try:
        for start in range(0, total, bs):
            end = min(start + bs, total)
            x_batch = x_all[start:end].to(device)
            conditioning_batch = (
                conditioning_all[start:end].to(device).view(-1)
                if conditioning_all is not None
                else None
            )

            if model_type in ["vit", "convnet"]:
                if coord_all is None:
                    raise ValueError(f"Model type '{model_type}' requires coordinate inputs, but dataset has no coords.")
                c_batch = coord_all[start:end].to(device)
                y_out = model(x_batch, c_batch[:, 0:1], c_batch[:, 1:2], conditioning_batch)
            else:
                y_out = model(x_batch, conditioning_batch)
            preds.append(y_out.cpu().view(-1))
    finally:
        if hook_handle:
            hook_handle.remove()

    y_pred = torch.cat(preds).numpy()
    y_true = ensure_tensor(ds.y).view(-1).numpy()
    if not np.all(np.isfinite(y_pred)):
        raise RuntimeError("Model predictions contain NaN or Inf.")
    feats = torch.cat(feats_list, dim=0).numpy() if feats_list else None

    target_metadata = _target_metadata_from(ds)
    model_target_metadata = _target_metadata_from(model)
    if model_target_metadata and target_metadata and model_target_metadata != target_metadata:
        raise ValueError("Model and dataset target metadata differ; refusing to mix target definitions.")
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    df.attrs["target_metadata"] = target_metadata
    df["sfr_true"] = _physical_sfr(y_true, target_metadata)
    df["sfr_pred"] = _physical_sfr(y_pred, target_metadata)

    baseline_data = getattr(ds, "y_baseline", None)
    if baseline_data is not None:
        df["y_baseline"] = ensure_tensor(baseline_data).view(-1).numpy()
        df.loc[np.isclose(df["y_baseline"], -100), "y_baseline"] = np.nan
        df["baseline_error"] = df["y_baseline"] - df["y_true"]
        df["sfr_baseline"] = _physical_sfr(df["y_baseline"].to_numpy(), target_metadata)

    if hasattr(ds, "time") and ds.time is not None:
        df["time"] = ensure_numpy(ds.time).reshape(-1)
    if hasattr(ds, "mdot_edd") and ds.mdot_edd is not None:
        df["mdot_edd"] = ensure_numpy(ds.mdot_edd).reshape(-1)

    df["error"] = df["y_pred"] - df["y_true"]

    if hasattr(ds, "group_names") and ds.group_names is not None and len(ds.group_names) == len(df):
        df["type"] = ds.group_names
    elif hasattr(ds, "type"):
        df["type"] = ds.type
    elif hasattr(ds, "groups") and ds.groups is not None:
        df["type"] = "unknown_group"
    else:
        df["type"] = "test_set"

    del x_all, conditioning_all, coord_all
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    if return_feats:
        return df, feats, x_data
    return df


def save_to_h5(path, df, feats, x_data, ds_type):
    with h5py.File(path, "w") as f:
        f.attrs["dataset_type"] = ds_type
        for key, value in _target_metadata_from(df).items():
            f.attrs[f"target_{key}"] = value
        f.create_dataset("x", data=x_data, compression="gzip")
        f.create_dataset("y_true", data=df["y_true"].values)
        f.create_dataset("y_pred", data=df["y_pred"].values)
        for column in ("sfr_true", "sfr_pred", "sfr_baseline"):
            if column in df.columns:
                f.create_dataset(column, data=df[column].values)
        if "type" in df.columns:
            dt = h5py.special_dtype(vlen=str)
            f.create_dataset("label", data=df["type"].values.astype(object), dtype=dt)
        if feats is not None:
            f.create_dataset("latent", data=feats)
        if "y_baseline" in df.columns:
            f.create_dataset("y_baseline", data=df["y_baseline"].values)
        if "time" in df.columns:
            f.create_dataset("time", data=df["time"].values)
        if "mdot_edd" in df.columns:
            f.create_dataset("mdot_edd", data=df["mdot_edd"].values)


# ==========================================
# 7. Visualization
# ==========================================
COLORS = ["#4B8D69", "#55658B", "#c1c16f", "#685587", "#B353CB", "#D78E5A", "#F06969"]


def save_plot(plt_obj, name, fig_dir, fmt="png"):
    path = fig_dir / f"{name}.{fmt}"
    plt_obj.savefig(path, dpi=150, bbox_inches="tight")


def plot_scatter(df, title, fig_dir, limit_range=True):
    baseline_col = get_baseline_column(df)
    has_baseline = baseline_col is not None
    cols_to_check = ["y_true", "y_pred"]
    if has_baseline:
        cols_to_check.append(baseline_col)

    vmin = df[cols_to_check].min().min() - 0.5
    vmax = df[cols_to_check].max().max() + 0.5

    def _plot_base(ax, show_baseline=False):
        ax.plot([vmin, vmax], [vmin, vmax], "r--", lw=1.5, label="Ideal", zorder=5)
        ax.plot(
            [vmin, vmax], [vmin - 1, vmax - 1], "k--", lw=0.8,
            alpha=0.5, label=r"$\pm$1 target-transform unit"
        )
        ax.plot([vmin, vmax], [vmin + 1, vmax + 1], "k--", lw=0.8, alpha=0.5)

        groups = df["type"].unique()
        for i, g in enumerate(groups):
            sub = df[df["type"] == g]
            ax.scatter(
                sub["y_true"],
                sub["y_pred"],
                s=8,
                alpha=0.7,
                color=COLORS[i % len(COLORS)],
                label=f"{g}",
                edgecolors="none",
            )

        # Only plot valid baseline values to avoid alignment issues after NaN filtering.
        if show_baseline and baseline_col is not None:
            valid_mask = df[baseline_col].notna()
            ax.scatter(
                df.loc[valid_mask, "y_true"],
                df.loc[valid_mask, baseline_col],
                s=15,
                marker="x",
                alpha=0.4,
                color="gray",
                label=get_baseline_label(df),
                zorder=1,
            )

        ax.set_xlabel(f"True {get_target_label(df)}")
        ax.set_ylabel(f"Predicted {get_target_label(df)}")
        if limit_range:
            ax.set_xlim(vmin, vmax)
            ax.set_ylim(vmin, vmax)
        ax.grid(True, linestyle=":", alpha=0.3)

    if has_baseline:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=120, sharex=True, sharey=True)
        _plot_base(axes[0], show_baseline=False)
        axes[0].set_title(f"{title} (Model Only)")
        _plot_base(axes[1], show_baseline=True)
        axes[1].set_title(f"{title} (With Baseline)")
        axes[0].legend(loc="upper left")
        axes[1].legend(loc="upper left")
        fig.tight_layout()
        save_plot(plt, f"scatter_{title.replace(' ', '_').lower()}", fig_dir)
        plt.show()
    else:
        plt.figure(figsize=(6, 6), dpi=120)
        ax = plt.gca()
        _plot_base(ax, show_baseline=False)
        ax.set_title(title)
        ax.legend(loc="upper left")
        save_plot(plt, f"scatter_{title.replace(' ', '_').lower()}", fig_dir)
        plt.show()


def plot_error_dist(df, title, fig_dir, model_label="Model"):
    plt.figure(figsize=(8, 5), dpi=120)
    plot_data = []
    tmp = df[["error", "type"]].copy()
    tmp["Method"] = model_label
    tmp.rename(columns={"error": "Error"}, inplace=True)
    plot_data.append(tmp)
    baseline_error_col = get_baseline_error_column(df)
    if baseline_error_col is not None:
        tmp2 = df[[baseline_error_col, "type"]].dropna().copy()
        tmp2["Method"] = get_baseline_label(df).replace("Baseline ", "")
        tmp2.rename(columns={baseline_error_col: "Error"}, inplace=True)
        plot_data.append(tmp2)
    final_df = pd.concat(plot_data)
    split = final_df["Method"].nunique() == 2
    sns.violinplot(data=final_df, x="type", y="Error", hue="Method", split=split, inner="quartile", palette="muted")
    plt.axhline(0, color="r", linestyle="--", alpha=0.5)
    plt.axhline(1, color="gray", linestyle=":", alpha=0.5)
    plt.axhline(-1, color="gray", linestyle=":", alpha=0.5)
    plt.title(f"Error Distribution: {title}")
    plt.ylabel("Error (target-transform units)")
    plt.xticks(rotation=45, ha="right")
    save_plot(plt, f"error_dist_{title.replace(' ', '_').lower()}", fig_dir)
    plt.show()


def plot_kde_contour(df, title, fig_dir):
    plt.figure(figsize=(6, 6), dpi=120)
    sns.kdeplot(
        x=df["y_true"],
        y=df["y_pred"],
        hue=df["type"],
        fill=True,
        alpha=0.3,
        levels=5,
        palette=COLORS[: len(df["type"].unique())],
    )
    vmin, vmax = df["y_true"].min(), df["y_true"].max()
    plt.plot([vmin, vmax], [vmin, vmax], "r--", lw=1)
    plt.title(f"Density: {title}")
    save_plot(plt, f"kde_{title.replace(' ', '_').lower()}", fig_dir)
    plt.show()


def plot_mass_evolution(df, title, fig_dir, model_label="Model"):
    """
    Backward-compatible wrapper.
    Older notebooks call this function for BH growth analysis; in the SFR
    pipeline we reinterpret it as SFR history versus time.
    """
    return plot_sfr_history(df, title, fig_dir, model_label=model_label)


def _sfr_series(df, physical_column, transformed_column):
    if physical_column in df.columns:
        return df[physical_column].to_numpy(dtype=float)
    return _physical_sfr(df[transformed_column].to_numpy(dtype=float), _target_metadata_from(df))


def _cumulative_trapezoid(values, times, time_unit_to_years):
    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)
    if len(values) != len(times):
        raise ValueError("SFR and time arrays must have the same length.")
    if len(values) == 0:
        return values
    dt = np.diff(times)
    if np.any(~np.isfinite(times)) or np.any(dt < 0):
        raise ValueError("Times must be finite and sorted in non-decreasing order.")
    increments = 0.5 * (values[:-1] + values[1:]) * dt * time_unit_to_years
    return np.concatenate(([0.0], np.cumsum(increments)))


def plot_cumulative_sfr(df, title, fig_dir, model_label="Model", sfr_unit_scale=1e9):
    """
    Plot cumulative stellar mass formed from the reconstructed SFR history.

    True/prediction/KS are instantaneous SFR. The plotted cumulative mass is
    derived with trapezoidal time integration; it is not a separately trained target.
    `time` is assumed to be in Gyr, hence the default 1e9 yr/Gyr factor.
    """
    if "time" not in df.columns:
        return

    df_plot = df.sort_values("time").copy()
    times = df_plot["time"].values
    sfr_true = _sfr_series(df_plot, "sfr_true", "y_true")
    sfr_pred = _sfr_series(df_plot, "sfr_pred", "y_pred")
    mass_true = _cumulative_trapezoid(sfr_true, times, sfr_unit_scale)
    mass_pred = _cumulative_trapezoid(sfr_pred, times, sfr_unit_scale)

    plt.figure(figsize=(10, 5), dpi=120)
    plt.plot(times, mass_true, "k-", lw=2, label="True", alpha=0.8)
    plt.plot(times, mass_pred, "r--", lw=2, label=f"Predicted ({model_label})", alpha=0.8)

    baseline_col = get_baseline_column(df_plot)
    if baseline_col is not None:
        sfr_baseline = _sfr_series(df_plot, "sfr_baseline", baseline_col)
        mass_baseline = _cumulative_trapezoid(sfr_baseline, times, sfr_unit_scale)
        plt.plot(times, mass_baseline, "g:", lw=1.5, label=get_baseline_label(df_plot), alpha=0.7)

    plt.yscale("log")

    plt.title(f"Cumulative Stellar Mass Formed: {title}")
    plt.xlabel("Time (Gyr)")
    plt.ylabel(r"$M_{\star,\mathrm{formed}}$ [$M_{\odot}$]")
    plt.legend()
    save_plot(plt, f"cum_sfr_{title.replace(' ', '_').lower()}", fig_dir)
    plt.show()


def plot_sfr_history(df, title, fig_dir, model_label="Model"):
    """
    Plot the reconstructed SFR history versus time.

    Values are inverted with the transform metadata saved beside the cache.
    `symlog` is used so genuine zero-SFR snapshots remain visible.
    """
    if "time" not in df.columns:
        return

    df_plot = df.sort_values("time").copy()
    times = df_plot["time"].values
    sfr_true = _sfr_series(df_plot, "sfr_true", "y_true")
    sfr_pred = _sfr_series(df_plot, "sfr_pred", "y_pred")

    plt.figure(figsize=(10, 5), dpi=120)
    plt.plot(times, sfr_true, "k-", lw=2, label="True", alpha=0.8)
    plt.plot(times, sfr_pred, "r--", lw=2, label=f"Predicted ({model_label})", alpha=0.8)

    baseline_col = get_baseline_column(df_plot)
    if baseline_col is not None:
        sfr_baseline = _sfr_series(df_plot, "sfr_baseline", baseline_col)
        plt.plot(times, sfr_baseline, "g:", lw=1.5, label=get_baseline_label(df_plot), alpha=0.7)

    positive = np.concatenate([sfr_true[sfr_true > 0], sfr_pred[sfr_pred > 0]])
    linthresh = max(float(np.nanmin(positive)) if positive.size else 1e-9, 1e-12)
    plt.yscale("symlog", linthresh=linthresh)
    plt.title(f"SFR History: {title}")
    plt.xlabel("Time (Gyr)")
    plt.ylabel(r"$\mathrm{SFR}$ [$M_{\odot}\,\mathrm{yr}^{-1}$]")
    plt.legend()
    save_plot(plt, f"sfr_history_{title.replace(' ', '_').lower()}", fig_dir)
    plt.show()


def get_metrics_summary(y_true, y_pred, name=""):
    try:
        from sklearn.metrics import r2_score
    except:

        def r2_score(t, p):
            residual = np.sum((t - p) ** 2)
            total = np.sum((t - np.mean(t)) ** 2)
            if total == 0:
                return 1.0 if residual == 0 else 0.0
            return 1 - residual / total

    d = y_true - y_pred
    return {
        "Dataset": name,
        "RMSE": np.sqrt(np.mean(d**2)),
        "MAE": np.mean(np.abs(d)),
        "R2": r2_score(y_true, y_pred),
        "OutlierFrac (|target error|>1)": np.mean(np.abs(d) > 1.0),
    }
