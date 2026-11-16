import argparse
import sys
from pathlib import Path

import mne
import numpy as np
import torch
from scipy.signal import resample


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EDF = PROJECT_ROOT / "data/raw/chbmit/chb01/chb01_03.edf"
DEFAULT_CBRAMOD_DIR = PROJECT_ROOT / "CBraMod"

# Exact 16-channel order in the authors' CHB-MIT process2.py.
# MNE renames the FIRST of two identically named T8-P8 channels to T8-P8-0.
CHANNELS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP2-F8", "F8-T8", "T8-P8-0", "P8-O2",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
]


def load_eeg_tensor(edf_path: Path) -> torch.Tensor:
    """Return one 10-second EEG window shaped (1, 16, 10, 200)."""
    if not edf_path.is_file():
        raise FileNotFoundError(f"EDF recording not found: {edf_path}")

    raw = mne.io.read_raw_edf(str(edf_path), preload=False)
    sampling_rate = float(raw.info["sfreq"])
    print(f"Original sampling rate: {sampling_rate} Hz")
    print(f"Original channel count: {len(raw.ch_names)}")

    missing = [name for name in CHANNELS if name not in raw.ch_names]
    if missing:
        raise ValueError(f"Missing required bipolar channels: {missing}")

    n_original_samples = round(10 * sampling_rate)
    if raw.n_times < n_original_samples:
        raise ValueError("The recording is shorter than 10 seconds.")

    # Read only the first 10 seconds, in the order expected by CBraMod.
    # MNE returns EEG amplitudes in volts.
    data_volts = raw.get_data(picks=CHANNELS, start=0, stop=n_original_samples)

    # The authors' CHB-MIT loader uses scipy.signal.resample to get 2000
    # samples per 10-second window and divides microvolt-valued signals by 100.
    data_microvolts = data_volts * 1e6
    data_200hz = resample(data_microvolts, 2000, axis=1)
    data_normalized = data_200hz / 100.0

    print("EEG array shape:", data_normalized.shape)  # (16, 2000)
    x = torch.from_numpy(np.asarray(data_normalized, dtype=np.float32))
    x = x.reshape(1, 16, 10, 200)
    print("CBraMod input shape:", tuple(x.shape))
    print("Input values finite:", bool(torch.isfinite(x).all()))
    return x


def test_pretrained_backbone(x: torch.Tensor, cbramod_dir: Path) -> None:
    """Load the official checkpoint and extract learned EEG features on CPU."""
    code_file = cbramod_dir / "models/cbramod.py"
    checkpoint = cbramod_dir / "pretrained_weights/pretrained_weights.pth"
    if not code_file.is_file():
        raise FileNotFoundError(
            f"CBraMod source not found: {code_file}\n"
            "Pass --cbramod-dir pointing to your cloned official repository."
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Pretrained checkpoint not found: {checkpoint}\n"
            "Download pretrained_weights.pth from the authors' Hugging Face page."
        )

    sys.path.insert(0, str(cbramod_dir))
    from models.cbramod import CBraMod  # noqa: E402

    device = torch.device("cpu")
    model = CBraMod().to(device)
    state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    # Match the official quick_example.py: output patch representations,
    # not the masked-reconstruction projection.
    model.proj_out = torch.nn.Identity()
    model.eval()

    with torch.inference_mode():
        features = model(x.to(device))

    print("Backbone feature shape:", tuple(features.shape))  # (1, 16, 10, 200)
    print("Backbone features finite:", bool(torch.isfinite(features).all()))
    print("NOTE: These are embeddings, NOT seizure/non-seizure predictions.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edf", type=Path, default=DEFAULT_EDF)
    parser.add_argument("--cbramod-dir", type=Path, default=DEFAULT_CBRAMOD_DIR)
    parser.add_argument(
        "--preprocess-only", action="store_true",
        help="Check the EEG tensor without loading CBraMod or its checkpoint.",
    )
    args = parser.parse_args()

    x = load_eeg_tensor(args.edf.expanduser().resolve())
    if not args.preprocess_only:
        test_pretrained_backbone(x, args.cbramod_dir.expanduser().resolve())


if __name__ == "__main__":
    main()
