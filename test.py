#!/usr/bin/env python
import numpy
from sagemaker.sklearn import SKLearn, SKLearnModel
from lib.role import SagemakerRoleConfig
from lib.model import (
    ModelHandler,
    PretrainedConfiguration,
    TrainingConfiguration,
    DeploymentConfiguration,
)

test_data = numpy.array([1, 2, 3, 4, 5, 10, 11, 22])
test_data = test_data.reshape(-1, 1)

# define clustering algorithm
training_config = TrainingConfiguration(
    model_class=SKLearn,
    entry_point="clustering.py",
    source_dir="./code/lib/model_hosting",
    instance_type="ml.m5.large",
    py_version="py3",
    framework_version="0.23-1",
)

pretrained_config = PretrainedConfiguration(
    model_class=SKLearnModel,
    model_data="s3://sagemaker-us-east-1-399446234556/clustering.tar.gz",
    entry_point="clustering.py",
    source_dir="./code/lib/model_hosting",
    framework_version="0.23-1",
)

deployment_config = DeploymentConfiguration(
    memory=4096,
    concurrency=10,
)

model_handler = ModelHandler.create(SagemakerRoleConfig())

# deploy models
predictor = model_handler.train_and_deploy(training_config, deployment_config)
pretrained_predictor = model_handler.deploy_pretrained(
    pretrained_config, deployment_config
)

# predict
model_handler.predict(predictor, test_data)
model_handler.predict(pretrained_predictor, test_data)

# teardown
model_handler.tear_down(predictor)
model_handler.tear_down(pretrained_predictor)
