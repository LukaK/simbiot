#!/usr/bin/env python
import numpy
from sagemaker.sklearn import SKLearn

# from sagemaker.sklearn import SKLearnModel
from lib.model import (
    ModelHandler,
    ModelConfiguration,
    PretrainedConfiguration,
    TrainingConfiguration,
    DeploymentConfiguration,
)

clustering_model_location = "s3://sagemaker-us-east-1-399446234556/clustering.tar.gz"

test_data = numpy.array([1, 2, 3, 4, 5, 10, 11, 22])
test_data = test_data.reshape(-1, 1)

# define clustering algorithm
training_config = TrainingConfiguration(
    entry_point="clustering.py",
    source_dir="./code/lib/model_hosting",
    instance_type="ml.m5.large",
    py_version="py3",
    framework_version="0.23-1",
)

pretrained_config = PretrainedConfiguration(
    model_data="s3://sagemaker-us-east-1-399446234556/clustering.tar.gz",
    entry_point="clustering.py",
    source_dir="./code/lib/model_hosting",
    framework_version="0.23-1",
)

deployment_config = DeploymentConfiguration(
    memory=4096,
    concurrency=10,
)

model_config = ModelConfiguration(
    training=training_config, pretrained=pretrained_config, deployment=deployment_config
)
model_handler = ModelHandler()
model_handler.initialize()
deployed_model = model_handler.train_and_deploy(SKLearn, deployment_config)
# deployed_model = model_handler.deploy_pretrained(SKLearnModel, deployment_config)
predictions = model_handler.predict(deployed_model, test_data)
model_handler.tear_down(deployed_model)
