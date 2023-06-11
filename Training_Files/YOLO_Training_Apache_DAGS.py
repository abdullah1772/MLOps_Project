from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2021, 1, 1),
    'email': ['your_email@your_domain.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG("yolo_training",
         default_args=default_args,
         schedule_interval=timedelta(days=1)) as dag:

    t0 = BashOperator(
        task_id="install_wandb",
        bash_command="pip install wandb",
    )

    t0_1 = BashOperator(
        task_id="wandb_login",
        bash_command="wandb login Wandb_API",
    )

    t1 = BashOperator(
        task_id="train_form",
        bash_command="python3 path/to/yolo/train.py --batch 2 --epochs 3 --data path/to/Forms_dataset/data.yaml --weights path/to/model_weights/YOLOv5_Form.pt --project path/tog/model_weights --name YOLOv5_Form --entity Wandb_Username --project Your_proj_name",
    )

    t2 = BashOperator(
        task_id="train_rating",
        bash_command="python3 path/to/yolo/train.py --batch 2 --epochs 3 --data path/to/Rating_dataset/data.yaml --weights path/to/model_weights/YOLOv5_scale.pt --project path/to/model_weights --name YOLOv5_rating --entity Wandb_Username --project Your_proj_name",
    )

    t3 = BashOperator(
        task_id="train_textbox",
        bash_command="python3 path/to/yolo/train.py --batch 2 --epochs 3 --data path/to/Text_box_dataset/data.yaml --weights path/to/model_weights/YOLOv5_line.pt --project path/to/model_weights --name YOLOv5_textbox --entity Wandb_Username --project Your_proj_name",
    )

    t0 >> t0_1 >> t1 >> t2 >> t3  # defines the order of the tasks
