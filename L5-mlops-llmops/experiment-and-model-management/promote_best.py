# promote_best.py
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()
MODEL_NAME = "iris-rf-classifier"

# 1) 在实验里按 accuracy 降序检索，取最优 run
exp = client.get_experiment_by_name("iris-rf-tuning")
runs = client.search_runs(
    experiment_ids=[exp.experiment_id],
    order_by=["metrics.accuracy DESC"],
    max_results=1,
)
best = runs[0]
best_acc = best.data.metrics["accuracy"]
print(f"[best] run_id={best.info.run_id} accuracy={best_acc:.4f}")

# 2) 把最优 run 的模型注册进 Registry（首次会创建 Registered Model，产生 v1）
model_uri = f"runs:/{best.info.run_id}/model"
mv = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
print(f"[registered] name={MODEL_NAME} version={mv.version}")

# 3) 阶段晋升：把这个版本推到 Production，并归档其它旧 Production 版本
client.transition_model_version_stage(
    name=MODEL_NAME,
    version=mv.version,
    stage="Production",
    archive_existing_versions=True,  # 旧的 Production 自动转 Archived，保证线上唯一
)
print(f"[promoted] {MODEL_NAME} v{mv.version} -> Production")

# 4) 验证：按阶段加载「当前生产模型」，这就是线上该用的版本
prod_model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/Production")
print("[verify] loaded Production model:", type(prod_model).__name__)
