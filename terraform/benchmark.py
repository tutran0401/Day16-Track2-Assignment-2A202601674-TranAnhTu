import json
import time
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


DATA_PATH = Path.home() / "ml-benchmark" / "creditcard.csv"
RESULT_PATH = Path.home() / "ml-benchmark" / "benchmark_result.json"
RANDOM_STATE = 42


def main() -> None:
    load_started = time.perf_counter()
    data = pd.read_csv(DATA_PATH)
    load_seconds = time.perf_counter() - load_started

    x = data.drop(columns="Class")
    y = data["Class"]
    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    x_valid, x_test, y_valid, y_test = train_test_split(
        x_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
    )

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )

    training_started = time.perf_counter()
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    training_seconds = time.perf_counter() - training_started

    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    one_row = x_test.iloc[[0]]
    model.predict_proba(one_row)  # warm up
    latency_started = time.perf_counter_ns()
    model.predict_proba(one_row)
    latency_ms = (time.perf_counter_ns() - latency_started) / 1_000_000

    thousand_rows = x_test.iloc[:1000]
    throughput_started = time.perf_counter()
    model.predict_proba(thousand_rows)
    thousand_seconds = time.perf_counter() - throughput_started

    results = {
        "instance": {"type": "c7i-flex.large", "vcpus": 2, "memory_gib": 4},
        "dataset": {
            "name": "mlg-ulb/creditcardfraud",
            "rows": int(len(data)),
            "features": int(x.shape[1]),
            "train_rows": int(len(x_train)),
            "validation_rows": int(len(x_valid)),
            "test_rows": int(len(x_test)),
        },
        "model": "LightGBM LGBMClassifier",
        "load_data_seconds": round(load_seconds, 6),
        "training_seconds": round(training_seconds, 6),
        "best_iteration": int(model.best_iteration_),
        "auc_roc": round(float(roc_auc_score(y_test, probabilities)), 6),
        "accuracy": round(float(accuracy_score(y_test, predictions)), 6),
        "f1_score": round(float(f1_score(y_test, predictions)), 6),
        "precision": round(float(precision_score(y_test, predictions)), 6),
        "recall": round(float(recall_score(y_test, predictions)), 6),
        "inference_latency_1_row_ms": round(latency_ms, 6),
        "inference_1000_rows_seconds": round(thousand_seconds, 6),
        "inference_throughput_rows_per_second": round(1000 / thousand_seconds, 2),
    }

    RESULT_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"\nSaved results to {RESULT_PATH}")


if __name__ == "__main__":
    main()
