# End to End MLOPS

## 1. Setup Folder Structure

### GitHub Repo

1. Create a repossitory
2. Clone repo to local 

### VSCode

1. Create a template.py

The template.py creates a folder structure template of the entire project.

```python

import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')


project_name = "mlProject"


list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/entity/config_entity.py",
    f"src/{project_name}/constants/__init__.py",
    "config/config.yaml",
    "params.yaml",
    "schema.yaml",
    "main.py",
    "app.py",
    "Dockerfile",
    "requirements.txt",
    "setup.py",
    "research/trials.ipynb",
    "templates/index.html",
    "test.py"
]


for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)


    if filedir !="":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")


    else:
        logging.info(f"{filename} is already exists")
```

### Terminal

ctrl +shift + `

```
python template.py
```

## 2. Install Packages and Implement Setup

1. On requirements.txt add the following packages:

```
pandas 
mlflow==2.2.2
notebook
numpy
scikit-learn
matplotlib
python-box==6.0.2
pyYAML
tqdm
ensure==1.0.2
joblib
types-PyYAML
Flask
Flask-Cors
-e .
```

The setup.py looks for the contructors (__init__.py) in the directory, and installs the folder as a local package.

### VSCode

```python
import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


__version__ = "0.0.0"

REPO_NAME = "end-to-end-mlops"
AUTHOR_USER_NAME = "sclauguico"
SRC_REPO = "mlProject"
AUTHOR_EMAIL = "sclauguico@gmail.com"


setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A small python package for ml app",
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)
```

### Terminal

```
pip install -r requirements.txt
```

## 3. Logging and Exception

### VSCode

src constructor setups the custom logging

```python
import os
import sys
import logging

logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

log_dir = "logs"
log_filepath = os.path.join(log_dir,"running_logs.log")
os.makedirs(log_dir, exist_ok=True)


logging.basicConfig(
    level= logging.INFO,
    format= logging_str,

    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("mlProjectLogger")

```

On main.py for example, we can then call the logger module


```python
from mlProject import logger

logger.info("Welcome to the custom logging!")
```

or

```python
from src.mlProject import logger

logger.info("Welcome to the custom logging!")
```

### Terminal

Running main.py creates a logs folder with running_logs.log file that 
stores all information abobut the logs.

```
python main.py
```

### VSCode

utils > common.py is used for defining all the most frequently used functions to avoid defining them in all scripts that need to run them

```python

import os
from box.exceptions import BoxValueError
import yaml
from mlProject import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any



@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")


@ensure_annotations
def save_json(path: Path, data: dict):
    """save json data

    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"json file saved at: {path}")




@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """load json files data

    Args:
        path (Path): path to json file

    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    with open(path) as f:
        content = json.load(f)

    logger.info(f"json file loaded succesfully from: {path}")
    return ConfigBox(content)


@ensure_annotations
def save_bin(data: Any, path: Path):
    """save binary file

    Args:
        data (Any): data to be saved as binary
        path (Path): path to binary file
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """load binary data

    Args:
        path (Path): path to binary file

    Returns:
        Any: object stored in the file
    """
    data = joblib.load(path)
    logger.info(f"binary file loaded from: {path}")
    return data



@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"


```


### Notebook

In trials.py, the use of box and ensure_annotations are demonstrated and explained.

Box is a Python library that makes inaccessible keys safe to access as an attribute.

ensure_annotations is a decorator that ensures that the correct data type is used. 

```jupyter

trials.ipynb
```

## Data Ingestion Workflow

### VSCode

1. Update config.yaml
    ```
    artifacts_root: artifacts

    data_ingestion:
        root_dir: artifacts/data_ingestion
        source_url: https://github.com/sclauguico/end-to-end-mlops/raw/main/data/winequality-data.zip
        local_data_file: artifacts/data_ingestion/data.zip
        unzip_dir: artifacts/data_ingestion
    ```
2. Update schema.yaml
3. Update params.yaml
4. Update the entity

    Try in the 01_data_ingestion.ipynb and copy to entity > config_entity.py to update the entity:

    ```python
        from dataclasses import dataclass
        from pathlib import Path


        @dataclass(frozen=True)
        class DataIngestionConfig:
            root_dir: Path
            source_URL: str
            local_data_file: Path
            unzip_dir: Path
    ```
5. Update the configuration manager in src config

    In the constant constructor, define the following paths:
    ```
        from pathlib import Path

        CONFIG_FILE_PATH = Path("config/config.yaml")
        PARAMS_FILE_PATH = Path("params.yaml")
        SCHEMA_FILE_PATH = Path("schema.yaml")
    ```

    In the 01_data_ingestion.ipynb, try updating the config manager and and then copy to src > components > config > configuration.py:

    ```python

        from mlProject.constants import *
        from mlProject.utils.common import read_yaml, create_directories

        class ConfigurationManager:
            def __init__(
                self,
                config_filepath = CONFIG_FILE_PATH,
                params_filepath = PARAMS_FILE_PATH,
                schema_filepath = SCHEMA_FILE_PATH):

                self.config = read_yaml(config_filepath)
                self.params = read_yaml(params_filepath)
                self.schema = read_yaml(schema_filepath)

                create_directories([self.config.artifacts_root])


            
            def get_data_ingestion_config(self) -> DataIngestionConfig:
                config = self.config.data_ingestion

                create_directories([config.root_dir])

                data_ingestion_config = DataIngestionConfig(
                    root_dir=config.root_dir,
                    source_URL=config.source_URL,
                    local_data_file=config.local_data_file,
                    unzip_dir=config.unzip_dir 
                )

                return data_ingestion_config
    ```
6. Update the components

    Try in 01_data_ingestion.ipynb and then copy and implement to a new file under components > data_ingestion.ipynb

    ```python
            import os
            import urllib.request as request
            import zipfile
            from mlProject import logger
            from mlProject.utils.common import get_size

            class DataIngestion:
            def __init__(self, config: DataIngestionConfig):
                self.config = config


            def download_file(self):
                if not os.path.exists(self.config.local_data_file):
                    filename, headers = request.urlretrieve(
                        url = self.config.source_URL,
                        filename = self.config.local_data_file
                    )
                    logger.info(f"{filename} download! with following info: \n{headers}")
                else:
                    logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")


            def extract_zip_file(self):
                """
                zip_file_path: str
                Extracts the zip file into the data directory
                Function returns None
                """
                unzip_path = self.config.unzip_dir
                os.makedirs(unzip_path, exist_ok=True)
                with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
                    zip_ref.extractall(unzip_path)
        
    ```
7. Update the pipeline

    On 01_data_ingestion.ipynb,
    ```python
            try:
                config = ConfigurationManager()
                data_ingestion_config = config.get_data_ingestion_config()
                data_ingestion = DataIngestion(config=data_ingestion_config)
                data_ingestion.download_file()
                data_ingestion.extract_zip_file()
            
            except Exception as e:
                raise e
    ```

    Create a new file under the pipeline folder called stage_01_data.ingestion.py

    ```python
            from mlProject.config.configuration import ConfigurationManager
            from mlProject.components.data_ingestion import DataIngestion
            from mlProject import logger

            STAGE_NAME = "Data Ingestion stage"

            class DataIngestionTrainingPipeline:
                def __init__(self):
                    pass

                def main(self):
                    config = ConfigurationManager()
                    data_ingestion_config = config.get_data_ingestion_config()
                    data_ingestion = DataIngestion(config=data_ingestion_config)
                    data_ingestion.download_file()
                    data_ingestion.extract_zip_file()

                
            if __name__ == '__main__':
                try:
                    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
                    obj = DataIngestionTrainingPipeline()
                    obj.main()
                    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
                except Exception as e:
                    logger.exception(e)
            raise e
    ```

    Add the following on params.yaml and schema.yaml so they will not be empty

    ```python
        key: val    
    ```
8. Update the main.py
    On main.py call the stage 1: data ingestion

    ```python
        from mlProject import logger
        from mlProject.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline

        STAGE_NAME = "Data Ingestion stage"
        try:
            logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
            data_ingestion = DataIngestionTrainingPipeline()
            data_ingestion.main()
            logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
        except Exception as e:
                logger.exception(e)
                raise e
    ```

9. Update the app.py

### GUI

1. Delete the artifacts folder

### Terminal 

1. 
```
python main.py
```


## Data Validation Workflow

### VSCode

1. Update config.yaml
    ```
        artifacts_root: artifacts

        data_validation:
        root_dir: artifacts/data_validation
        unzip_data_dir: artifacts/data_ingestion/winequality-red.csv
        STATUS_FILE: artifacts/data_validation/status.txt
    ```
2. Update schema.yaml
    ```
        COLUMNS:
        fixed acidity: float64
        volatile acidity: float64
        citric acid: float64
        residual sugar: float64
        chlorides: float64
        free sulfur dioxide: float64
        total sulfur dioxide: float64
        density: float64
        pH: float64
        sulphates: float64
        alcohol: float64
        quality: int64


        TARGET_COLUMN:
        name: quality
    ```
3. Update params.yaml
4. Update the entity

    Try in the 02_data_validation.ipynb and copy to entity > config_entity.py to update the entity:

    ```python
        from dataclasses import dataclass
        from pathlib import Path


        @dataclass(frozen=True)
        class DataValidationConfig:
            root_dir: Path
            STATUS_FILE: str
            unzip_data_dir: Path
            all_schema:dict
    ```
5. Update the configuration manager in src config

    In the constant constructor, define the following paths:
    ```
        from pathlib import Path

        CONFIG_FILE_PATH = Path("config/config.yaml")
        PARAMS_FILE_PATH = Path("params.yaml")
        SCHEMA_FILE_PATH = Path("schema.yaml")
    ```

    In the 02_data_validation.ipynb, try updating the config manager and and then copy to src > components > config > configuration.py:

    ```python

        from mlProject.constants import *
        from mlProject.utils.common import read_yaml, create_directories


            def __init__(
                self,
                config_filepath = CONFIG_FILE_PATH,
                params_filepath = PARAMS_FILE_PATH,
                schema_filepath = SCHEMA_FILE_PATH):

                self.config = read_yaml(config_filepath)
                self.params = read_yaml(params_filepath)
                self.schema = read_yaml(schema_filepath)

                create_directories([self.config.artifacts_root])


            
            def get_data_validation_config(self) -> DataValidationConfig:
                config = self.config.data_validation
                schema = self.schema.COLUMNS

                create_directories([config.root_dir])

                data_validation_config = DataValidationConfig(
                    root_dir=config.root_dir,
                    STATUS_FILE=config.STATUS_FILE,
                    unzip_data_dir = config.unzip_data_dir,
                    all_schema=schema,
                )

                return data_validation_config
    ```
6. Update the components

    Try in 02_data_validation.ipynb and then copy and implement to a new file under components > data_validation.ipynb

    ```python
            import os
            from mlProject import logger
            from mlProject.entity.config_entity import DataValidationConfig
            import pandas as pd


            class DataValidation:
                def __init__(self, config: DataValidationConfig):
                    self.config = config


                def validate_all_columns(self)-> bool:
                    try:
                        validation_status = None

                        data = pd.read_csv(self.config.unzip_data_dir)
                        all_cols = list(data.columns)

                        all_schema = self.config.all_schema.keys()

                        
                        for col in all_cols:
                            if col not in all_schema:
                                validation_status = False
                                with open(self.config.STATUS_FILE, 'w') as f:
                                    f.write(f"Validation status: {validation_status}")
                            else:
                                validation_status = True
                                with open(self.config.STATUS_FILE, 'w') as f:
                                    f.write(f"Validation status: {validation_status}")

                        return validation_status
                    
                    except Exception as e:
                        raise e

    ```
7. Update the pipeline

    On 02-data_validation.ipynb
    ```python
        try:
            config = ConfigurationManager()
            data_validation_config = config.get_data_validation_config()
            data_validation = DataValidation(config=data_validation_config)
            data_validation.validate_all_columns()
        except Exception as e:
            raise e
    ```

    Create a new file under the pipeline folder called stage_02_data.validation.py

    ```python
        from mlProject.config.configuration import ConfigurationManager
        from mlProject.components.data_validation import DataValidation
        from mlProject import logger


        STAGE_NAME = "Data Validation stage"

        class DataValidationTrainingPipeline:
            def __init__(self):
                pass

            def main(self):
                config = ConfigurationManager()
                data_validation_config = config.get_data_validation_config()
                data_validation = DataValidation(config=data_validation_config)
                data_validation.validate_all_columns()


        if __name__ == '__main__':
            try:
                logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
                obj = DataValidationTrainingPipeline()
                obj.main()
                logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
            except Exception as e:
                logger.exception(e)
                raise e


    ```

    Add the following on params.yaml and schema.yaml so they will not be empty

    ```python
        key: val    
    ```
8. Update the main.py
    On main.py call the stage 1: data ingestion

    ```python
        from mlProject import logger
        from mlProject.pipeline.stage_01_data_validation import DataValidationTrainingPipeline

        STAGE_NAME = "Data Validation stage"
        try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
        data_ingestion = DataValidationTrainingPipeline()
        data_ingestion.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
        except Exception as e:
                logger.exception(e)
                raise e
    ```

9. Update the app.py

### GUI

1. Delete the artifacts folder

### Terminal 

1. 
```
python main.py
```