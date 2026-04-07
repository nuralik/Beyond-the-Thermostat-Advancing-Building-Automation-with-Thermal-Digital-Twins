from pathlib import Path
import pandas as pd

# Folder containing Sample00, Sample01, ..., Sample05
ROOT = Path(r"./data")
RESAMPLE_RULE = "5min"

GROUPS = {
    "lab1": ["Sample00", "Sample01"],
    "lab2": ["Sample02", "Sample03", "Sample04", "Sample05"],
}

VARIABLES = ["Temperature", "Humidity"]


def suffix(sample_name: str) -> str:
    return sample_name[-2:]


def read_time(sample_dir: Path, sfx: str) -> pd.Series:
    df = pd.read_excel(sample_dir / f"Time{sfx}.xlsx")
    return pd.to_datetime(df["tin"])


def build_group(group_name: str, sample_names: list[str]):
    all_times = []
    all_data = {var: [] for var in VARIABLES}

    for sample_name in sample_names:
        sample_dir = ROOT / sample_name
        sfx = suffix(sample_name)

        t = read_time(sample_dir, sfx)
        all_times.append(t)

        for var in VARIABLES:
            df = pd.read_excel(sample_dir / f"{var}{sfx}.xlsx")
            if len(df) != len(t):
                raise ValueError(
                    f"Row mismatch in {sample_dir / f'{var}{sfx}.xlsx'} "
                    f"({len(df)} rows) vs Time{sfx}.xlsx ({len(t)} rows)"
                )
            all_data[var].append(df)

    # Concatenate
    time_all = pd.concat(all_times, ignore_index=True)
    data_all = {
        var: pd.concat(parts, ignore_index=True)
        for var, parts in all_data.items()
    }

    # Sort by time before resampling
    order = time_all.sort_values(kind="stable").index
    time_all = pd.Series(time_all.iloc[order].values, name="tin").reset_index(drop=True)
    for var in VARIABLES:
        data_all[var] = data_all[var].iloc[order].reset_index(drop=True)

    # Output folder
    out_dir = ROOT / group_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save common 5-minute time axis
    dummy = pd.Series(1, index=pd.DatetimeIndex(time_all))
    time_5min = dummy.resample(RESAMPLE_RULE).mean().index
    pd.DataFrame({"tin": time_5min}).to_excel(out_dir / "Time_5min.xlsx", index=False)

    # Save 5-minute averages
    for var in VARIABLES:
        df = data_all[var].copy()
        df.index = pd.DatetimeIndex(time_all)
        df_5min = df.resample(RESAMPLE_RULE).mean()
        df_5min.to_excel(out_dir / f"{var}_5min.xlsx", index=False)

    print(f"{group_name} done -> {out_dir}")


if __name__ == "__main__":
    for group_name, sample_names in GROUPS.items():
        build_group(group_name, sample_names)

    print("All done.")