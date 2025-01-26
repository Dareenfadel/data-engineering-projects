# To be able to import your function, you need to add the src/ directory to the Python path.
import pandas as pd
# For Label Encoding
from sklearn import preprocessing
from sqlalchemy import create_engine
from functions import extract_clean,transform,load_to_postgres

from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from fintech_dashboard import create_dashboard


# def extract_clean(filename):
#     df = pd.read_csv(filename)
#     df = clean(df)
#     df.to_csv('/opt/airflow/data/fintech_clean.csv',index=True)
#     print('loaded after cleaning succesfully')

# def transform(filename):
#     df = pd.read_csv(filename)
#     df =encode_and_normalize(df)
#     try:
#         df.to_csv('/opt/airflow/data/fintech_transformed.csv',index=False, mode='x')
#         print('loaded after cleaning succesfully')
#     except FileExistsError:
#         print('file already exists')

# def load_to_csv(df,filename):
#     df.to_csv(filename,index=False)
#     print('loaded succesfully')
    
# def load_to_postgres(filename): 
#     df = pd.read_csv(filename)
#     engine = create_engine('postgresql://root:root@pgdatabase:5432/fintech_etl')
#     if(engine.connect()):
#         print('connected succesfully')
#     else:
#         print('failed to connect')
#     df.to_sql(name = 'fintech_db',con = engine,if_exists='replace')

# def create_dashboard_task(filename):
#     df = pd.read_csv(filename)
#     create_dashboard(df)

# Define the DAG
default_args = {
    "owner": "data_engineering_team",
    "depends_on_past": False,
    'start_date': days_ago(2),
    "retries": 1,
}

dag = DAG(
    'fintech_etl_pipeline',
    default_args=default_args,
    description='fintech etl pipeline',
)

with DAG(
    dag_id = 'fintech_etl_pipeline',
    schedule_interval = '@once', # could be @daily, @hourly, etc or a cron expression '* * * * *'
    default_args = default_args,
    tags = ['fintech-pipeline'],
)as dag:
    # Define the tasks
    extract_clean_task = PythonOperator(
        task_id = 'extract_clean',
        python_callable = extract_clean,
        op_kwargs = {
            'filename': '/opt/airflow/data/fintech.csv'
        }
    )

    transform_task = PythonOperator(
        task_id = 'tranform',
        python_callable = transform,
        op_kwargs = {
            'filename': '/opt/airflow/data/fintech_clean.csv'
        }
    )

    load_to_postgres_task = PythonOperator(
        task_id = 'load_to_postgres',
        python_callable = load_to_postgres,
        op_kwargs = {
            'filename': '/opt/airflow/data/fintech_transformed.csv'
        }
    )
    run_dashboard=PythonOperator(
        task_id = 'create_dashboard',
        python_callable = create_dashboard,
        op_kwargs = {
            'filename': '/opt/airflow/data/fintech_clean.csv'
        }
    )

    # Define the task dependencies
    extract_clean_task >> transform_task >> load_to_postgres_task >>run_dashboard