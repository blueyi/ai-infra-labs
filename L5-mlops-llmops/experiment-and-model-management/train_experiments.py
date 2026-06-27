# train_experiments.py
import json
import pathlib
import sys

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

RUNS_DIR = pathlib.Path("mlruns_lite")
EXPERIMENT = "iris-rf-tuning"


def log_run(params: dict, acc: float) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    exp_dir = RUNS_DIR / EXPERIMENT
    exp_dir.mkdir(exist_ok=True)
    run_id = f"run_{len(list(exp_dir.glob('*.json')))}"
    payload = {"params": params, "accuracy": acc}
    (exp_dir / f"{run_id}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"params={params} -> accuracy={acc:.4f}  (logged to {exp_dir / f'{run_id}.json'})")


def main() -> None:
    # Prefer real MLflow when Python has lzma (pyenv builds often lack it)
    try:
        import lzma  # noqa: F401
        import mlflow
        import mlflow.sklearn

        mlflow.set_experiment(EXPERIMENT)
        use_mlflow = True
    except Exception as e:
        print(f"[warn] MLflow unavailable ({e}); using file-based mlruns_lite/")
        use_mlflow = False

    X, y = load_iris(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

    param_grid = [
        {"n_estimators": 10, "max_depth": 2},
        {"n_estimators": 50, "max_depth": 3},
        {"n_estimators": 100, "max_depth": 5},
        {"n_estimators": 200, "max_depth": None},
    ]

    for params in param_grid:
        clf = RandomForestClassifier(random_state=42, **params)
        clf.fit(X_tr, y_tr)
        acc = accuracy_score(y_te, clf.predict(X_te))

        if use_mlflow:
            import mlflow
            import mlflow.sklearn

            with mlflow.start_run():
                mlflow.log_params(params)
                mlflow.log_metric("accuracy", acc)
                mlflow.sklearn.log_model(clf, artifact_path="model")
                print(f"params={params} -> accuracy={acc:.4f}")
        else:
            log_run(params, acc)

    print(f"[done] {len(param_grid)} runs logged")


if __name__ == "__main__":
    main()
