import pendulum
from airflow.sdk import dag, task

@dag(schedule=None, start_date=pendulum.datetime(2024, 1, 1, tz="UTC"), catchup=False)
def hello():
    @task
    def say(msg: str) -> str:
        print(msg)
        return msg.upper()

    @task
    def shout(x: str):
        print(x)

    shout(say("hello airflow"))

hello()