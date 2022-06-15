#!/usr/bin/env python
import pytest


def test__deployment_configuration_initialization():
    from lib.model import DeploymentConfiguration

    DeploymentConfiguration(memory=4096, concurrency=10)


def test__training_configuration_initialization():
    from lib.model import TrainingConfiguration

    TrainingConfiguration(
        entry_point="test_file.py",
        source_dir="test/directory",
        instance_type="test instance",
        py_version="py3",
        framework_version="testversion",
    )


def test__pretrained_configuration_initialization():
    from lib.model import PretrainedConfiguration

    PretrainedConfiguration(
        model_data="test",
        entry_point="test",
        source_dir="test",
        framework_version="test",
    )


def test__model_configuration_initialization():
    from lib.model import (
        PretrainedConfiguration,
        TrainingConfiguration,
        DeploymentConfiguration,
        ModelConfiguration,
    )

    training_config = TrainingConfiguration(
        entry_point="test_file.py",
        source_dir="test/directory",
        instance_type="test instance",
        py_version="py3",
        framework_version="testversion",
    )
    pretrained_config = PretrainedConfiguration(
        model_data="test",
        entry_point="test",
        source_dir="test",
        framework_version="test",
    )
    deployment_config = DeploymentConfiguration(memory=4096, concurrency=10)
    ModelConfiguration(
        pretrained=pretrained_config,
        training=training_config,
        deployment=deployment_config,
    )
