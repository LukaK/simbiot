#!/usr/bin/env python
import numpy
from typing import Protocol
from .model_refactored import ModelConfiguration, DeployedModel


class ModelHandler(Protocol):
    def train_and_deploy(
        self, model_configuration: ModelConfiguration
    ) -> DeployedModel:
        ...

    def deploy_pretrained(
        self, model_configuration: ModelConfiguration
    ) -> DeployedModel:
        ...

    def tear_down(self, deployed_model: DeployedModel) -> None:
        ...

    def predict(
        self, deployed_model: DeployedModel, data: numpy.ndarray
    ) -> numpy.ndarray:
        ...
