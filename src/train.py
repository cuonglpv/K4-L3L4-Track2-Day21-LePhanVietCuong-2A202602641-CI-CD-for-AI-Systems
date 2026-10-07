import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
import warnings
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

# Nguong chat luong cua lab nay la f1_score, KHONG phai accuracy.
# Ly do: bo du lieu Adult co ty le lop 75/25. Mot mo hinh doan bua
# "thu nhap thap" cho moi mau da dat accuracy 0.75 ma khong hoc duoc gi.
F1_THRESHOLD = 0.65
POSITIVE_RATE_BASELINE = 0.248
POSITIVE_RATE_DRIFT_THRESHOLD = 0.05


def train(
    params: dict,
    data_path: str = "data/train_batch1.csv",
    eval_path: str = "data/holdout.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.

    Tham so:
        params     : dict chua cac sieu tham so cho GradientBoostingClassifier.
        data_path  : duong dan den file du lieu huan luyen.
        eval_path  : duong dan den file du lieu danh gia (holdout).

    Tra ve:
        f1 (float): diem F1 cua lop duong (thu nhap > 50K) tren tap holdout.
    """

    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    positive_rate = float(y_train.mean())
    positive_rate_drift = abs(positive_rate - POSITIVE_RATE_BASELINE)
    drift_warning = positive_rate_drift > POSITIVE_RATE_DRIFT_THRESHOLD
    if drift_warning:
        warnings.warn(
            "DATA DRIFT WARNING: positive-class rate "
            f"{positive_rate:.2%} differs from the {POSITIVE_RATE_BASELINE:.1%} "
            f"baseline by more than {POSITIVE_RATE_DRIFT_THRESHOLD:.0%}.",
            stacklevel=2,
        )

    with mlflow.start_run():

        mlflow.log_params(params)

        model = GradientBoostingClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_eval)
        f1 = f1_score(y_eval, preds)
        acc = accuracy_score(y_eval, preds)
        report_by_class = classification_report(
            y_eval, preds, output_dict=True, zero_division=0
        )
        matrix = confusion_matrix(y_eval, preds).tolist()
        precision_0 = report_by_class["0"]["precision"]
        recall_0 = report_by_class["0"]["recall"]
        precision_1 = report_by_class["1"]["precision"]
        recall_1 = report_by_class["1"]["recall"]

        mlflow.log_metrics(
            {
                "f1_score": f1,
                "accuracy": acc,
                "precision_class_0": precision_0,
                "recall_class_0": recall_0,
                "precision_class_1": precision_1,
                "recall_class_1": recall_1,
                "train_positive_rate": positive_rate,
                "positive_rate_drift": positive_rate_drift,
            }
        )
        mlflow.set_tag("data_drift_warning", str(drift_warning).lower())
        mlflow.sklearn.log_model(
            model,
            "model",
            pip_requirements=["scikit-learn==1.4.2"],
        )

        print(
            f"F1: {f1:.4f} | Accuracy: {acc:.4f} | "
            f"Precision(1): {precision_1:.4f} | Recall(1): {recall_1:.4f}"
        )
        print(
            f"Train positive rate: {positive_rate:.2%} | "
            f"Drift warning: {drift_warning}"
        )

        os.makedirs("outputs", exist_ok=True)
        evaluation = {
            "confusion_matrix": matrix,
            "per_class": {
                "0": {"precision": precision_0, "recall": recall_0},
                "1": {"precision": precision_1, "recall": recall_1},
            },
            "train_positive_rate": positive_rate,
            "positive_rate_baseline": POSITIVE_RATE_BASELINE,
            "positive_rate_drift": positive_rate_drift,
            "data_drift_warning": drift_warning,
        }
        with open("outputs/report.json", "w") as f:
            json.dump(
                {
                    "f1_score": f1,
                    "accuracy": acc,
                    "precision": precision_1,
                    "recall": recall_1,
                    **evaluation,
                },
                f,
            )
        with open("outputs/evaluation.json", "w") as f:
            json.dump(evaluation, f)
        mlflow.log_artifact("outputs/evaluation.json")

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.joblib")

    return float(f1)


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
