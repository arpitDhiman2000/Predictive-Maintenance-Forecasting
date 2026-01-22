import re
from pathlib import Path

import pandas as pd
from pymongo import MongoClient


MONGO_URI = "mongodb+srv://arpitdhiman:Ad%401234@cluster0.yovyubm.mongodb.net/?appName=Cluster0"
DB_NAME = "PMF_RawData"
SERIES_COL = "cmaps_series"
RUL_COL = "cmaps_rul"

DATA_DIR = Path("notebook/CMAPSSData")
CHUNK_SIZE = 50000            


FILE_RE = re.compile(r"^(train|test|RUL)_(FD00[1-5])\.txt$", re.IGNORECASE)

CMAPSS_COLS = (
    ["engine_id", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

OP_SETTINGS_COLS = [f"op_setting_{i}" for i in range(1, 4)]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]


def parse_filename(path: Path):
    m = FILE_RE.match(path.name)
    if not m:
        return None
    split = m.group(1).lower()
    dataset_id = m.group(2).upper()
    return split, dataset_id


def read_series(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    df = df.iloc[:, :len(CMAPSS_COLS)]
    df.columns = CMAPSS_COLS
    df["engine_id"] = df["engine_id"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    return df


def chunked(iterable, size: int):
    for i in range(0, len(iterable), size):
        yield iterable[i:i+size]


def main():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    series_col = db[SERIES_COL]
    rul_col = db[RUL_COL]

    # One-time load: drop old data first (prevents duplicates)
    series_col.drop()
    rul_col.drop()

    # Keep test engine_id order per dataset for correct RUL mapping
    test_engine_ids_order = {}

    files = sorted([p for p in DATA_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".txt"])

    for path in files:
        parsed = parse_filename(path)
        if not parsed:
            continue

        split, dataset_id = parsed

        if split in ("train", "test"):
            df = read_series(path)

            # Build documents
            df["dataset_id"] = dataset_id
            df["split"] = split
            docs = df[
                        ["dataset_id", "split", "engine_id", "cycle"] + OP_SETTINGS_COLS + SENSOR_COLS
                    ].to_dict(orient="records")

            # Insert in chunks
            for batch in chunked(docs, CHUNK_SIZE):
                series_col.insert_many(batch, ordered=False)

            print(f"[OK] Inserted {len(docs)} rows: {path.name}")

            if split == "test":
                test_engine_ids_order[dataset_id] = df["engine_id"].drop_duplicates().tolist()

        elif split == "rul":
            rul_values = pd.read_csv(path, header=None)[0].astype(int).tolist()

            # Ensure test engine_ids are known (read test file if needed)
            if dataset_id not in test_engine_ids_order:
                test_path = DATA_DIR / f"test_{dataset_id}.txt"
                test_df = read_series(test_path)
                test_engine_ids_order[dataset_id] = test_df["engine_id"].drop_duplicates().tolist()

            engine_ids = test_engine_ids_order[dataset_id]
            if len(engine_ids) != len(rul_values):
                raise ValueError(f"Mismatch for {dataset_id}: test engine_ids={len(engine_ids)} vs RUL lines={len(rul_values)}")

            rul_docs = [{"dataset_id": dataset_id, "engine_id": int(u), "rul": int(r)} for u, r in zip(engine_ids, rul_values)]
            for batch in chunked(rul_docs, CHUNK_SIZE):
                rul_col.insert_many(batch, ordered=False)

            print(f"[OK] Inserted {len(rul_docs)} RUL rows: {path.name}")

    # indexes for faster queries
    series_col.create_index([("dataset_id", 1), ("split", 1), ("engine_id", 1), ("cycle", 1)])
    rul_col.create_index([("dataset_id", 1), ("engine_id", 1)])

    print("Done.")


if __name__ == "__main__":
    main()
