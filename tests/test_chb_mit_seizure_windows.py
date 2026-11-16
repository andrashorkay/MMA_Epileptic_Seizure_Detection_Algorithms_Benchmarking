
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

# Exact 16-channel order used by the authors.
# MNE renames the first duplicated T8-P8 channel to T8-P8-0.
CHANNELS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP2-F8", "F8-T8", "T8-P8-0", "P8-O2",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
]

WINDOW_SECONDS = 10
TARGET_HZ = 200

# Seizure in chb01_03.edf: 2996–3036 seconds
WINDOWS = {
    "non-seizure": 1000,
    "seizure": 3000,
}


def prepare_window(raw, channels, start_time):
    """Return an EEG tensor of shape (1, 16, 10, 200)."""

    original_hz = float(raw.info["sfreq"])

    start_sample = round(start_time * original_hz)
    stop_sample = round(
        (start_time + WINDOW_SECONDS) * original_hz
    )

    if start_sample < 0 or stop_sample > raw.n_times:
        raise ValueError("Window exceeds recording duration.")

    missing = [
        ch for ch in channels if ch not in raw.ch_names
    ]
    if missing:
        raise ValueError(f"Missing EEG channels: {missing}")

    # Extract the window in the required channel order.
    # MNE returns EEG values in volts.
    data = raw.get_data(
        picks=channels,
        start=start_sample,
        stop=stop_sample,
    )

    # Convert volts to microvolts.
    data = data * 1e6

    # Resample from 256 Hz to 200 Hz.
    data = resample(
        data,
        WINDOW_SECONDS * TARGET_HZ,
        axis=1,
    )

    # Normalise using the authors' 100 µV scale.
    data = data / 100.0

    # Reshape into one-second patches.
    data = data.reshape(16, 10, 200)

    # Add the batch dimension.
    x = torch.from_numpy(
        np.asarray(data, dtype=np.float32)
    ).unsqueeze(0)

    return x


def test_pretrained_backbone(windows, cbramod_dir):
    """Run both EEG windows through the pretrained CBraMod backbone."""

    code_file = cbramod_dir / "models/cbramod.py"
    checkpoint = (
        cbramod_dir
        / "pretrained_weights/pretrained_weights.pth"
    )

    if not code_file.is_file():
        raise FileNotFoundError(
            f"CBraMod source not found: {code_file}"
        )

    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}"
        )

    sys.path.insert(0, str(cbramod_dir))

    from models.cbramod import CBraMod

    device = torch.device("cpu")

    model = CBraMod().to(device)

    state_dict = torch.load(
        checkpoint,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(state_dict)

    # Extract patch embeddings rather than reconstructed EEG.
    model.proj_out = torch.nn.Identity()
    model.eval()

    with torch.inference_mode():
        for label, x in windows.items():

            features = model(x.to(device))

            print(
                f"{label} feature shape:",
                tuple(features.shape),
            )

            print(
                f"{label} features finite:",
                bool(torch.isfinite(features).all()),
            )

    print(
        "\nNote: These are EEG embeddings, "
        "not seizure probabilities."
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--edf",
        type=Path,
        default=DEFAULT_EDF,
    )

    parser.add_argument(
        "--cbramod-dir",
        type=Path,
        default=DEFAULT_CBRAMOD_DIR,
    )

    parser.add_argument(
        "--preprocess-only",
        action="store_true",
    )

    args = parser.parse_args()

    edf_path = args.edf.expanduser().resolve()

    if not edf_path.is_file():
        raise FileNotFoundError(edf_path)

    # Load the recording.
    raw = mne.io.read_raw_edf(
        str(edf_path),
        preload=False,
    )

    print("Recording:", edf_path.name)
    print("Sampling frequency:", raw.info["sfreq"])
    print("Original channel count:", len(raw.ch_names))
    print("Seizure interval: 2996–3036 seconds\n")

    windows = {}

    # Extract and preprocess both windows.
    for label, start_time in WINDOWS.items():

        x = prepare_window(
            raw,
            CHANNELS,
            start_time,
        )

        windows[label] = x

        print(f"{label} window:", start_time)
        print(f"{label} shape:", tuple(x.shape))

        print(
            f"{label} range:",
            x.min().item(),
            x.max().item(),
        )

        print(
            f"{label} finite:",
            bool(torch.isfinite(x).all()),
        )

        print()

    # Run the model unless preprocessing-only was requested.
    if not args.preprocess_only:
        test_pretrained_backbone(
            windows,
            args.cbramod_dir.expanduser().resolve(),
        )


if __name__ == "__main__":
    main()