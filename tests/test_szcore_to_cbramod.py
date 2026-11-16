"""CPU-only tests for converting SzCORE referential EEG to CBraMod bipolar EEG.

Run from project root:
    python3 tests/test_szcore_to_cbramod.py

Optionally test a real *SzCORE-formatted* EDF file (NOT raw CHB-MIT):
    python3 tests/test_szcore_to_cbramod.py --edf path/to/szcore_recording.edf
"""

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from szcore_cbramod_adapter import (  # noqa: E402
    SZCORE_CHANNELS,
    CBRAMOD_CHANNEL_NAMES,
    CBRAMOD_DERIVATIONS,
    prepare_cbramod_window,
    szcore_to_cbramod_bipolar,
)


def run_synthetic_tests():
    rng = np.random.default_rng(2026)
    n_samples = 256 * 10

    # Construct physically plausible synthetic electrode signals in VOLTS.
    # A common offset is deliberately added and then removed by average ref.
    potentials = rng.normal(0, 20e-6, size=(19, n_samples))
    global_offset = 50e-6 * np.sin(np.linspace(0, 20, n_samples))
    unreferenced = potentials + global_offset[None, :]
    szcore_car = unreferenced - unreferenced.mean(axis=0, keepdims=True)

    bipolar = szcore_to_cbramod_bipolar(szcore_car, SZCORE_CHANNELS)
    assert bipolar.shape == (16, n_samples)

    # Check every output against the DIRECT underlying electrode difference.
    electrode_idx = {name.upper(): i for i, name in enumerate(SZCORE_CHANNELS)}
    expected = np.stack(
        [
            potentials[electrode_idx[a.upper()]] - potentials[electrode_idx[b.upper()]]
            for _, a, b in CBRAMOD_DERIVATIONS
        ],
        axis=0,
    )
    np.testing.assert_allclose(bipolar, expected, rtol=1e-10, atol=1e-15)
    print("[PASS] All 16 bipolar derivations match direct electrode differences")

    # A shared reference must cancel even if it changes every time sample.
    random_common_reference = rng.normal(0, 100e-6, size=n_samples)
    shifted = szcore_car + random_common_reference[None, :]
    np.testing.assert_allclose(
        szcore_to_cbramod_bipolar(shifted, SZCORE_CHANNELS),
        bipolar,
        rtol=1e-10,
        atol=1e-15,
    )
    print("[PASS] Common-reference signal cancels at every time sample")

    # Correct selection must not depend on the order of the 19 input channels.
    order = rng.permutation(19)
    shuffled_names = [SZCORE_CHANNELS[i] for i in order]
    np.testing.assert_allclose(
        szcore_to_cbramod_bipolar(szcore_car[order], shuffled_names),
        bipolar,
        rtol=1e-10,
        atol=1e-15,
    )
    print("[PASS] Input channel reordering does not change output")

    # Check full format adaptation: 256 -> 200 Hz and 10 x one-second patches.
    x = prepare_cbramod_window(szcore_car, SZCORE_CHANNELS)
    assert x.shape == (1, 16, 10, 200)
    assert x.dtype == np.float32
    assert np.isfinite(x).all()
    print("[PASS] CBraMod tensor has shape (1, 16, 10, 200), float32")

    # Missing electrodes must fail instead of silently making up data.
    try:
        szcore_to_cbramod_bipolar(szcore_car[:18], SZCORE_CHANNELS[:18])
    except ValueError as exc:
        assert "Missing" in str(exc)
    else:
        raise AssertionError("Missing channel was accepted")
    print("[PASS] Missing required EEG electrodes are rejected")

    # Original CHB-MIT EDF's bipolar naming is NOT valid SzCORE input.
    try:
        szcore_to_cbramod_bipolar(
            np.zeros((16, n_samples)), CBRAMOD_CHANNEL_NAMES
        )
    except ValueError as exc:
        assert "Not a complete" in str(exc)
    else:
        raise AssertionError("Bipolar CHB-MIT input was incorrectly accepted")
    print("[PASS] Original CHB-MIT bipolar data are rejected as SzCORE input")


def run_real_edf_test(edf: Path, start: float):
    import mne

    raw = mne.io.read_raw_edf(str(edf), preload=False)
    hz = float(raw.info["sfreq"])
    if hz != 256.0:
        raise ValueError(f"Expected 256 Hz SzCORE EDF; found {hz} Hz")
    start_sample = round(start * hz)
    stop_sample = start_sample + 10 * 256
    if start_sample < 0 or stop_sample > raw.n_times:
        raise ValueError("Requested 10-second interval lies outside EDF recording")

    # Read only a 10-second window, including any extra EDF channels.
    samples_volts = raw.get_data(start=start_sample, stop=stop_sample)
    x = prepare_cbramod_window(samples_volts, raw.ch_names)
    print(f"[PASS] Real SzCORE EDF: {edf.name}")
    print(f"       Input channels: {len(raw.ch_names)}")
    print(f"       CBraMod input:  {x.shape}, {x.dtype}")
    print(f"       Normalized range: {x.min():.4f} to {x.max():.4f}")
    print("       This validates data format, not seizure prediction accuracy.")


def main():
    parser = argparse.ArgumentParser(description="SzCORE -> CBraMod CPU adapter test")
    parser.add_argument("--edf", type=Path, help="Optional genuine SzCORE 19-channel EDF")
    parser.add_argument("--start", type=float, default=0.0, help="Window start in seconds")
    args = parser.parse_args()

    print("Running synthetic SzCORE montage tests...\n")
    run_synthetic_tests()
    print("\nSynthetic conversion tests passed.")

    if args.edf is not None:
        print("\nChecking a real SzCORE-formatted EDF...")
        if not args.edf.is_file():
            parser.error(f"File not found: {args.edf}")
        run_real_edf_test(args.edf, args.start)
    else:
        print("\nNo real SzCORE EDF provided; synthetic tests only.")


if __name__ == "__main__":
    main()
