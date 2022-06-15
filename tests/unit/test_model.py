#!/usr/bin/env python
import pytest
from moto.core import patch_client


def test__deployment_configuration_initialization():
    from lib.model import DeploymentConfiguration

    DeploymentConfiguration(memory=4096, concurrency=10)


def test__training_configuration_initialization():
    from lib.model import TrainingConfiguration
    from sagemaker.sklearn import SKLearn

    TrainingConfiguration(
        model_class=SKLearn,
        entry_point="test_file.py",
        source_dir="test/directory",
        instance_type="test instance",
        py_version="py3",
        framework_version="testversion",
    )


def test__pretrained_configuration_initialization():
    from lib.model import PretrainedConfiguration
    from sagemaker.sklearn import SKLearnModel

    PretrainedConfiguration(
        model_class=SKLearnModel,
        model_data="test",
        entry_point="test",
        source_dir="test",
        framework_version="test",
    )


def test__create_model_handler(iam_client):
    from lib.role import RoleHandler, SagemakerRoleConfig
    from lib.model import ModelHandler

    patch_client(RoleHandler._iam_client)
    role_config = SagemakerRoleConfig(name="test_role")
    model_handler = ModelHandler.create(role_config)
