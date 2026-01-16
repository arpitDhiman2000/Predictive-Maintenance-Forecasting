import os
from pathlib import Path

project_dir = "src"

list_of_files = [

    f'{project_dir}/__init__.py',
    f'{project_dir}/components/data_ingestion.py',
    f'{project_dir}/components/data_validation.py',
    f'{project_dir}/components/data_transformation.py',
    f'{project_dir}/components/model_trainer.py',
    f'{project_dir}/components/model_evaluation.py',
    f'{project_dir}/components/model_pusher.py',
    f'{project_dir}/configuration/__init__.py',
    f'{project_dir}/configuration/postgre_connection.py',
    f'{project_dir}/configuration/aws_connection.py',
    f'{project_dir}/cloud_storage/__init__.py',
    f'{project_dir}/cloud_storage/aws_storage.py',
    f'{project_dir}/data_access/__init__.py',
    f'{project_dir}/data_access/data.py',
    f'{project_dir}/constants/__init__.py',
    f'{project_dir}/entity/__init__.py',
    f'{project_dir}/entity/config_entity.py',
    f'{project_dir}/entity/artifact_entity.py',
    f'{project_dir}/entity/estimator.py',
    f'{project_dir}/entity/s3_estimator.py',
    f'{project_dir}/exception/__init__.py',
    f'{project_dir}/logger/__init__.py',
    f'{project_dir}/pipline/__init__.py',
    f'{project_dir}/pipline/training_pipline.py',
    f'{project_dir}/pipline/prediction_pipline.py',
    f'{project_dir}/utils/__init__.py',
    f'{project_dir}/utils/main_utils.py',
    "app.py",
    "requirements.txt",
    "Dockerfile",
    ".dockerignore",
    "demo.py",
    "setup.py",
    "config/model.yaml",
    "config/schema.yaml",
    "pyproject.toml",

]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
    
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, 'w') as f:
            pass

    else:
        print(f"File already present at: {filepath}")