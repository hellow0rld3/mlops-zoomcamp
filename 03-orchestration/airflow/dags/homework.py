import pendulum
from airflow.sdk import dag
from airflow.providers.standard.operators.bash import BashOperator


@dag(
    schedule=None,    
    max_active_runs=1,
    default_args={"retries": 1, "retry_delay": pendulum.duration(minutes=1)},
    tags=["mlops-zoomcamp-homework"],
)
def homework_yellow_lr():
    check_mlflow = BashOperator(
        task_id="check_mlflow",
        bash_command="curl -sf http://127.0.0.1:5000/health",
    )

    train_model = BashOperator(
        task_id="train_model",
        cwd="/Users/cszczepanek/projects/mlops-zoomcamp/03-orchestration",
        bash_command="/Users/cszczepanek/projects/mlops-zoomcamp/.venv/bin/python homework.py",
    )

    check_mlflow >> train_model


homework_yellow_lr()