# ml/train.py
import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from src.ml.schema import FEATURES

DATA_PATH = "data/processed/ml_input_features.csv"
MODEL_PATH = "models/accept_model.joblib"
META_PATH = "models/accept_model_meta.json"


def evaluate(y_true, proba, threshold: float = 0.3) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "auc": float(roc_auc_score(y_true, proba)) if len(set(y_true)) > 1 else None,
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
        "classification_report": classification_report(
            y_true, pred, digits=3, zero_division=0
        ),
    }


def main(threshold: float = 0.3):
    # 데이터 로드
    df = pd.read_csv(DATA_PATH)

    # 필수 컬럼 체크
    missing_cols = [c for c in FEATURES + ["accept"] if c not in df.columns]
    if missing_cols:
        raise KeyError(f"CSV에 필요한 컬럼이 없습니다: {missing_cols}")

    # X/y 분리
    X = df[FEATURES].copy()
    y = df["accept"].astype(int).copy()

    # train/valid split
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if len(set(y)) > 1 else None,
    )

    # 모델 파이프라인
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )

    # 학습
    model.fit(X_train, y_train)

    # 검증(전체)
    valid_proba = model.predict_proba(X_valid)[:, 1]
    metrics_all = evaluate(y_valid, valid_proba, threshold=threshold)

    print("\n=== [VALID] Overall Metrics ===")
    print(f"threshold={metrics_all['threshold']}")
    print(f"AUC={metrics_all['auc'] if metrics_all['auc'] is not None else 'N/A'}")
    print(f"ACC={metrics_all['accuracy']:.3f}  F1={metrics_all['f1']:.3f}")
    print("Confusion Matrix:", metrics_all["confusion_matrix"])
    print("\nClassification Report:\n", metrics_all["classification_report"])

    # filter_level 별 성능(있는 경우)
    by_level = None
    if "filter_level" in X_valid.columns:
        print("\n=== [VALID] Metrics by filter_level ===")
        by_level = {}

        for level in sorted(X_valid["filter_level"].unique()):
            idx = X_valid["filter_level"] == level
            y_lv = y_valid[idx]
            p_lv = valid_proba[idx]

            m = evaluate(y_lv, p_lv, threshold=threshold)
            by_level[int(level)] = {
                "n": int(idx.sum()),
                "auc": m["auc"],
                "accuracy": m["accuracy"],
                "f1": m["f1"],
                "confusion_matrix": m["confusion_matrix"],
            }

            print(
                f"level={int(level)}  n={int(idx.sum())}  "
                f"AUC={m['auc'] if m['auc'] is not None else 'N/A'}  "
                f"ACC={m['accuracy']:.3f}  F1={m['f1']:.3f}"
            )

    # 모델 저장
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\n[OK] saved model: {MODEL_PATH}")

    # 메타 저장(재현/디버깅용)
    meta = {
        "data_path": DATA_PATH,
        "model_path": MODEL_PATH,
        "features": FEATURES,
        "n_rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "valid_rows": int(len(X_valid)),
        "accept_rate": float(y.mean()),
        "train_config": {
            "threshold": float(threshold),
            "model": "LogisticRegression",
            "max_iter": 2000,
            "class_weight": "balanced",
            "scaler": "StandardScaler",
            "random_state": 42,
            "test_size": 0.2,
        },
        "valid_metrics": metrics_all,
        "valid_by_filter_level": by_level,
    }

    os.makedirs(os.path.dirname(META_PATH), exist_ok=True)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[OK] saved meta: {META_PATH}")


if __name__ == "__main__":
    main(threshold=0.3)
