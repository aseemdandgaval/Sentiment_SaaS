import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.debugger import TensorBoardOutputConfig

boto_sess = boto3.Session(region_name="us-east-2")
sagemaker_sess = sagemaker.Session(boto_session=boto_sess)

def start_training():
    tensorboard_config = TensorBoardOutputConfig(
        s3_output_path="s3://your-bucket-name/tensorboard",
        container_local_output_path="/opt/ml/output/tensorboard"
    )

    estimator = PyTorch(
        entry_point="train.py",
        source_dir="training",
        role="your-arn-and-execution-role",
        framework_version="2.5.1",
        py_version="py311",
        instance_count=1,
        instance_type="ml.g5.xlarge",
        hyperparameters={
            "batch-size": 32,
            "epochs": 25
        },
        tensorboard_output_config=tensorboard_config,
    )

    try:
        print("Starting training...")
        estimator.fit({
            "training": "s3://your-bucket-name/dataset/train",
            "validation": "s3://your-bucket-name/dataset/dev",
            "test": "s3://your-bucket-name/dataset/test",
             })
    except Exception as e:
        print(e)
    
   

if __name__ == "__main__":
    start_training()
