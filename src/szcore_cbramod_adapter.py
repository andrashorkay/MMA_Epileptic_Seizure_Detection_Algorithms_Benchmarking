"""Adapt SzCORE's 19 referential EEG channels to CBraMod's 16 bipolar channels.

Input:  19 channels in the SzCORE standard (common-average-referenced; volts).
Output: 16 bipolar derivations in CBraMod CHB-MIT ordering.

Reference: https://epilepsybenchmarks.com/framework/
CBraMod: https://github.com/wjq-learning/CBraMod/blob/main/preprocessing/CHB-MIT/process2.py

This adapter does not modify SzCORE or train a seizure classifier.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np
from scipy.signal import resample


# Official SzCORE input order (19 common-average-referenced EEG channels).
SZCORE_CHANNELS = (
    "Fp1-Avg", "F3-Avg", "C3-Avg", "P3-Avg", "O1-Avg",
    "F7-Avg", "T3-Avg", "T5-Avg", "Fz-Avg", "Cz-Avg",
    "Pz-Avg", "Fp2-Avg", "F4-Avg", "C4-Avg", "P4-Avg",
    "O2-Avg", "F8-Avg", "T4-Avg", "T6-Avg",
)

# (CBraMod bipolar channel name, positive SzCORE electrode, negative SzCORE electrode).
# In SzCORE's 10-20 terminology: T3 = T7, T4 = T8, T5 = P7, T6 = P8.
CBRAMOD_DERIVATIONS = (
    ("FP1-F7", "Fp1-Avg", "F7-Avg"),
    ("F7-T7", "F7-Avg", "T3-Avg"),
    ("T7-P7", "T3-Avg", "T5-Avg"),
    ("P7-O1", "T5-Avg", "O1-Avg"),
    ("FP2-F8", "Fp2-Avg", "F8-Avg"),
    ("F8-T8", "F8-Avg", "T4-Avg"),
    ("T8-P8", "T4-Avg", "T6-Avg"),
    ("P8-O2", "T6-Avg", "O2-Avg"),
    ("FP1-F3", "Fp1-Avg", "F3-Avg"),
    ("F3-C3", "F3-Avg", "C3-Avg"),
    ("C3-P3", "C3-Avg", "P3-Avg"),
    ("P3-O1", "P3-Avg", "O1-Avg"),
    ("FP2-F4", "Fp2-Avg", "F4-Avg"),
    ("F4-C4", "F4-Avg", "C4-Avg"),
    ("C4-P4", "C4-Avg", "P4-Avg"),
    ("P4-O2", "P4-Avg", "O2-Avg"),
)

CBRAMOD_CHANNEL_NAMES = tuple(x[0] for x in CBRAMOD_DERIVATIONS)


def _key(name: str) -> str:
    return name.strip().upper()


def szcore_to_cbramod_bipolar(
    eeg_volts: np.ndarray,
    channel_names: Sequence[str] = SZCORE_CHANNELS,
) -> np.ndarray:
    """Convert SzCORE-style referential EEG to CBraMod bipolar derivations.

    Parameters
    ----------
    eeg_volts : array, shape (n_input_channels, n_samples)
        EEG measurements in volts, referenced to the SAME common average.
        Additional channels are allowed if they are separately named.
    channel_names : sequence of str
        Labels matching the first axis; channels may be reordered, but all 19
        canonical SzCORE channel labels must be present exactly once.

    Returns
    -------
    array, shape (16, n_samples)
        16 bipolar signals, still in volts, in the CBraMod CHB-MIT order.

    Notes
    -----
    Differences of channels with a shared reference cancel that reference:
    (Va - Vavg) - (Vb - Vavg) == Va - Vb.
    This works only if the inputs really are referential and consistently
    referenced; channel names alone cannot prove this.
    """
    eeg = np.asarray(eeg_volts)
    if eeg.ndim != 2:
        raise ValueError(f"Expected 2D (channels, samples), got shape {eeg.shape}")
    if eeg.shape[1] == 0:
        raise ValueError("EEG window contains no samples")
    if len(channel_names) != eeg.shape[0]:
        raise ValueError("Number of channel names does not match EEG channel axis")
    if not np.issubdtype(eeg.dtype, np.number) or not np.isfinite(eeg).all():
        raise ValueError("EEG samples must be finite numerical values")

    keys = [_key(name) for name in channel_names]
    duplicates = [name for name, count in Counter(keys).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate channel names: {duplicates}")

    name_to_idx = {name: i for i, name in enumerate(keys)}
    missing = [name for name in SZCORE_CHANNELS if _key(name) not in name_to_idx]
    if missing:
        raise ValueError(
            "Not a complete 19-channel SzCORE referential montage. "
            f"Missing: {missing}. Do NOT pass original CHB-MIT bipolar EDF here."
        )

    return np.stack(
        [
            eeg[name_to_idx[_key(positive)]] - eeg[name_to_idx[_key(negative)]]
            for _, positive, negative in CBRAMOD_DERIVATIONS
        ],
        axis=0,
    )


def prepare_cbramod_window(
    eeg_volts: np.ndarray,
    channel_names: Sequence[str] = SZCORE_CHANNELS,
    *,
    source_hz: int = 256,
) -> np.ndarray:
    """Return one 10-second CBraMod input as (1, 16, 10, 200) float32.

    This reproduces the channel difference, resampling and amplitude scaling
    used for the CHB-MIT pipeline. It is an input-format adapter, not a trained
    seizure detector. The EEG must arrive in volts from an MNE EDF reader.
    """
    if source_hz != 256:
        raise ValueError(f"Expected SzCORE 256-Hz input, got {source_hz} Hz")
    expected_samples = 10 * source_hz
    if np.asarray(eeg_volts).ndim != 2 or eeg_volts.shape[1] != expected_samples:
        raise ValueError(
            f"Expected precisely 10 seconds ({expected_samples} samples); "
            f"got {np.asarray(eeg_volts).shape}"
        )

    bipolar_volts = szcore_to_cbramod_bipolar(eeg_volts, channel_names)
    # Convert V -> microvolts; resample to 200 Hz; scale by 100 microvolts.
    bipolar_uv = bipolar_volts * 1e6
    downsampled = resample(bipolar_uv, 10 * 200, axis=-1)
    normalized = downsampled / 100.0
    return np.asarray(normalized, dtype=np.float32).reshape(1, 16, 10, 200)
