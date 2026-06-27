# promote_best.py — works with MLflow or mlruns_lite fallback
import json
import pathlib

RUNS_LITE = pathlib.Path("mlruns_lite/iris-rf-tuning")


def promote_lite():
    runs = []
    for p in RUNS_LITE.glob("run_*.json"):
        data = json.loads(p.read_text())
        runs.append((data["accuracy"], p))
    if not runs:
        raise SystemExit("no mlruns_lite runs; run train_experiments.py first")
    best_acc, best_path = max(runs, key=lambda x: x[0])
    print(f"[best] lite run={best_path.name} accuracy={best_acc:.4f}")
    registry = pathlib.Path("model_registry.json")
    registry.write_text(
        json.dumps({"model": "iris-rf-classifier", "version": 1, "stage": "Production", "accuracy": best_acc}, indent=2),
        encoding="utf-8",
    )
    print("[promoted] iris-rf-classifier v1 -> Production (lite registry)")
    print("[verify] loaded Production metadata from model_registry.json")


def promote_mlflow():
    import mlflow
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    MODEL_NAME = "iris-rf-classifier"
    exp = client.get_experiment_by_name("iris-rf-tuning")
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["metrics.accuracy DESC"],
        max_results=1,
    )
    best = runs[0]
    best_acc = best.data.metrics["accuracy"]
    print(f"[best] run_id={best.info.run_id} accuracy={best_acc:.4f}")
    model_uri = f"runs:/{best.info.run_id}/model"
    mv = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
    print(f"[registered] name={MODEL_NAME} version={mv.version}")
    client.transition_model_version_stage(
        name=MODEL_NAME, version=mv.version, stage="Production", archive_existing_versions=True,
    )
    print(f"[promoted] {MODEL_NAME} v{mv.version} -> Production")
    prod_model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/Production")
    print("[verify] loaded Production model:", type(prod_model).__name__)


if __name__ == "__main__":
    try:
        import lzma  # noqa: F401
        import mlflow  # noqa: F401
        promote_mlflow()
    except Exception as e:
        print(f"[warn] MLflow path unavailable ({e}); using mlruns_lite")
        promote_lite()
