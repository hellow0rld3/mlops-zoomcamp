import pendulum
from airflow.sdk import dag
from airflow.providers.standard.operators.bash import BashOperator


@dag(
    schedule="@monthly",
    start_date=pendulum.datetime(2024, 3, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 1, "retry_delay": pendulum.duration(minutes=1)},
    tags=["mlops-zoomcamp"],
)
def nyc_taxi_duration():
    check_mlflow = BashOperator(
        task_id="check_mlflow",
        bash_command="curl -sf http://127.0.0.1:5000/health",
    )

    train_model = BashOperator(
        task_id="train_model",
        cwd="/Users/cszczepanek/projects/mlops-zoomcamp/03-orchestration",
        bash_command=(
            "/Users/cszczepanek/projects/mlops-zoomcamp/.venv/bin/python duration_prediction.py"
            " --year {{ (logical_date - macros.dateutil.relativedelta.relativedelta(months=2)).year }}"
            " --month {{ (logical_date - macros.dateutil.relativedelta.relativedelta(months=2)).month }}"
        ),
    )

    check_mlflow >> train_model


nyc_taxi_duration()