from pathlib import Path

summary_path = Path(
    "data/raw/chbmit/chb01/chb01-summary.txt"
)

text = summary_path.read_text()

start = text.index("File Name: chb01_03.edf")
end = text.find("File Name:", start + 1)

if end == -1:
    end = len(text)

print(text[start:end])