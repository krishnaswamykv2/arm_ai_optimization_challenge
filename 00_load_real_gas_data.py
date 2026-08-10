"""
STEP: Load the real Gas Sensor Array Drift Dataset (Kaggle/UCI).

This dataset is NOT a plain CSV -- it's stored in libsvm-style format:
    <gas_class>;<concentration> 1:<val> 2:<val> ... 128:<val>

This script parses that format into a normal pandas table so it can be
used the same way as our synthetic hazard_dataset.csv.

BEFORE RUNNING: update DATA_FOLDER below to point to wherever you
extracted the downloaded dataset (the folder containing batch1.dat,
batch2.dat, etc.)
"""

import pandas as pd
import glob
import os

DATA_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "archive", "Dataset")  # relative to this script's location, works on any machine

def parse_dat_file(filepath, batch_number, max_debug_lines=3):
    """
    Real file format confirmed from actual data:
        <gas_class> 1:val 2:val 3:val ... 128:val
    Example:
        1 1:15596.162100 2:1.868245 3:2.371604 ...
    No semicolon, no separate concentration value on the line.
    """
    rows = []
    skipped = 0
    debug_shown = 0

    with open(filepath, "r") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            if debug_shown < max_debug_lines:
                print(f"  [DEBUG] Raw line {line_num}: {line[:120]}")
                debug_shown += 1

            try:
                parts = line.split(" ")
                gas_class = int(parts[0])

                row = {
                    "gas_class": gas_class,
                    "batch": batch_number
                }

                for pair in parts[1:]:
                    if ":" not in pair:
                        continue
                    idx, val = pair.split(":")
                    row[f"feature_{idx}"] = float(val)

                rows.append(row)
            except (ValueError, IndexError):
                skipped += 1
                if skipped <= 3:
                    print(f"  [SKIPPED] Line {line_num} did not match expected format: {line[:120]}")
                continue

    if skipped > 0:
        print(f"  -> Skipped {skipped} malformed lines in this file.")
    return rows


def load_all_batches(folder):
    all_rows = []
    dat_files = sorted(glob.glob(os.path.join(folder, "batch*.dat")))

    if not dat_files:
        raise FileNotFoundError(
            f"No batch*.dat files found in '{folder}'. "
            f"Check that DATA_FOLDER points to the extracted dataset folder."
        )

    for filepath in dat_files:
        # Extract batch number from filename, e.g. "batch3.dat" -> 3
        filename = os.path.basename(filepath)
        batch_num = int("".join(ch for ch in filename if ch.isdigit()))
        print(f"Parsing {filename} (batch {batch_num})...")
        all_rows.extend(parse_dat_file(filepath, batch_num))

    return pd.DataFrame(all_rows)


if __name__ == "__main__":
    df = load_all_batches(DATA_FOLDER)
    print(f"\nLoaded {len(df)} total measurements across all batches.")
    print(f"Columns: {df.shape[1]} (gas_class, concentration, batch, + 128 sensor features)")
    print("\nGas class distribution:")
    print(df["gas_class"].value_counts().sort_index())
    print("\nFirst few rows:")
    print(df.head())

    df.to_csv("real_gas_sensor_data.csv", index=False)
    print("\nSaved parsed data to real_gas_sensor_data.csv")
