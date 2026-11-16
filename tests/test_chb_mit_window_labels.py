
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
summary_path = ROOT / "data/raw/chbmit/chb01/chb01-summary.txt"

text = summary_path.read_text()

# Extract annotations for chb01_03.edf
block = text.split("File Name: chb01_03.edf", 1)[1]
block = block.split("File Name:", 1)[0]

starts = [
    int(x) for x in re.findall(
        r"Seizure(?: \d+)? Start Time:\s*(\d+) seconds",
        block
    )
]
ends = [
    int(x) for x in re.findall(
        r"Seizure(?: \d+)? End Time:\s*(\d+) seconds",
        block
    )
]

assert len(starts) == len(ends)
seizures = list(zip(starts, ends))

print("Seizure intervals:", seizures)


def authors_label(start, end, seizures):
    return int(any(
        start < seizure_start < end
        or start < seizure_end < end
        for seizure_start, seizure_end in seizures
    ))


def overlap_label(start, end, seizures):
    return int(any(
        start < seizure_end and end > seizure_start
        for seizure_start, seizure_end in seizures
    ))


print("\nWindow       Authors   Overlap")

for start in range(2980, 3050, 10):
    end = start + 10

    original = authors_label(start, end, seizures)
    corrected = overlap_label(start, end, seizures)

    print(
        f"{start:4d}–{end:4d}   "
        f"{original:^7d}   {corrected:^7d}"
    )
