from datetime import datetime, timedelta
import time
import requests

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy_operator import DummyOperator

SCRAPYD_ENDPOINT = "http://scrapyapp:6800/schedule.json"
SPIDER = "rightmove"
PROJECT = "rightmove_scraper"

def start_spider():
    payload = f"project={PROJECT}&spider={SPIDER}"
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }

    response = requests.request("POST", SCRAPYD_ENDPOINT, headers=headers, data=payload)

    print(response.text)
    if response.status_code == 200:
        print("Request successful")
        if response.json()['status'] == 'ok':
            print("Job started")
            job_id = response.json()['jobid']
            print("Job id:", job_id)
            return job_id
    else:
        print(response.text)
        raise ValueError("Request failed")

def cancel_spider(**kwargs):
    job_id = kwargs['ti'].xcom_pull(task_ids='start_spider')

    print(f"Cancelling job id: {job_id}")

    payload = f"project={PROJECT}&job={job_id}"
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'

    }

    url = "http://scrapyapp:6800/cancel.json"

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)
    if response.status_code == 200:
        print("Request successful")
        if response.json()['status'] == 'ok':
            print("Job cancelled successfully")
        else:
            print(response.text)
    else:
        print(response.text)
        raise ValueError("Request failed spider has not been canceled")

    return "Success"

def repeated_requests(**kwargs):
    end_time = datetime.now() + timedelta(seconds=90)

    url = f"http://scrapyapp:6800/listjobs.json?project={PROJECT}"

    payload = {}
    headers = {}

    job_id = kwargs['ti'].xcom_pull(task_ids='start_spider')

    while datetime.now() < end_time:
        response = requests.request("GET", url, headers=headers, data=payload)

        print(f"Response code: {response.status_code}")
        if response.status_code == 200:
            print("Request successful")
            if response.json()['status'] == 'ok':
                print("Scrapy status is okay")

                running_jobs = response.json()['running']

                if job_id in [job['id'] for job in running_jobs]:
                    print("Job is running")
            else:
                print(response.text)

        time.sleep(30) # wait for 30 seconds before next request
    return "Success"

default_args = {
    'owner': 'airflow_app',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'scrape_rightmove',
    default_args=default_args,
    description='DAG for making scraping rightmove',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2023, 1, 1),
    catchup=False,
    max_active_runs=1
)

start_task = DummyOperator(
    task_id='start',
    dag=dag
)

start_spider_task = PythonOperator(
    task_id='start_spider',
    python_callable=start_spider,
    dag=dag
)

periodic_requests = PythonOperator(
    task_id='periodic_requests',
    python_callable=repeated_requests,
    provide_context=True,
    dag=dag
)

cancel_spider_task = PythonOperator(
    task_id='cancel_spider',
    python_callable=cancel_spider,
    provide_context=True,
    dag=dag
)

end_task = DummyOperator(
    task_id='end',
    dag=dag
)

start_task >> start_spider_task >> periodic_requests >> cancel_spider_task >> end_task