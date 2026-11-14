from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

P = yaml.safe_load(open("configs/params.yaml"))["split"]
RAW = Path("data/raw/Telco-Customer-Churn.csv")
OUT = Path("data/batches")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RAW)
y, seed = P["target"], P["seed"]


def carve(frame, n):
    """frame'den stratified n satır ayır, (kalan, ayrılan) döndür."""
    rest, part = train_test_split(
        frame, test_size=n, stratify=frame[y], random_state=seed
    )
    return rest, part


rest, test = carve(df, P["test_rows"])
rest, batch3 = carve(rest, P["batch3_rows"])
batch1, batch2 = carve(rest, P["batch2_rows"])

for name, part in {
    "test_holdout": test,
    "batch_1": batch1,
    "batch_2": batch2,
    "batch_3": batch3,
}.items():
    part.to_csv(OUT / f"{name}.csv", index=False)
    print(f"{name}: {len(part)} satır, churn oranı {(part[y] == 'Yes').mean():.3f}")