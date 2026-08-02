# Instantaneous-SFR target and channel variants

The new training target is the sum of `Data-Set-11` in each raw snapshot. It
is stored as physical instantaneous SFR in `Msun/yr` and transformed for
training as:

```text
y = log10(1 + SFR / (1 Msun/yr))
```

The same transform is applied to the KS baseline. Zero-SFR snapshots are kept.
The older `Exp1_` caches and `Exp1_SFRConvNet` training path remain available
for reproducing historical cumulative-target runs, but they must not be mixed
with the instantaneous-SFR models or figures.

## Recommended eight-channel run

This is the default path. It excludes the field now named
`initial_stellar_density` (the old `dnewstar` channel):

```powershell
python raw_data_processing/galaxy_data.py --galaxy disk_galaxy_fiducial
python Experiment/prepare_data.py --only-exp1 --seed 42
python Experiment/run_train.py
```

The cache prefix is `Exp1_InstantSFR_NoInitialStellarDensity_`; the experiment
name is `Exp1_InstantSFR_NoInitialStellarDensity_SFRConvNet`.

## Retained-channel control

To keep `initial_stellar_density` as a ninth input channel while using the same
instantaneous target and split seed:

```powershell
python Experiment/prepare_data.py --only-exp1 --seed 42 --exclude-channels= --cache-tag InstantSFR
python Experiment/run_train.py Exp1_InstantSFR_SFRConvNet
```

The two variants have identical labels, baseline values, time stamps, and
train/validation/test membership. Only the selected input channel and its
normalization arrays differ.

Optuna defaults to the recommended eight-channel cache. The control can be
selected explicitly:

```powershell
python Experiment/run_optuna.py --data-exp Exp1_InstantSFR_
```

For raw-HDF5 analysis, always reuse both the training normalization and cached
metadata:

```python
from pathlib import Path
from VisualScripts.ResultAnalysis import analysis_utils as utils

prefix = "Exp1_InstantSFR_NoInitialStellarDensity_"
cached = utils.CachedPTDataset(Path(f".cache/{prefix}train.pt"))
norm = utils.load_norm_stats(Path(f".cache/{prefix}norm.npz"))

raw = utils.HDF5Dataset(
    hdf5_conf,
    mean=norm["mean"],
    std=norm["std"],
    reference_dataset=cached,
)
```

Model loading now rejects mismatched channel lists and mismatched target
semantics instead of silently analyzing the wrong cache/model combination.
