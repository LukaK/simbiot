#!/usr/bin/env python
from __future__ import annotations

import pickle  # nosec
from dataclasses import dataclass

import numpy
from sagemaker.deserializers import BytesDeserializer
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker.serializers import IdentitySerializer
from sagemaker.serverless import ServerlessInferenceConfig

from .logger import logger
from .role import RoleHandler, SagemakerRoleConfig


@dataclass(frozen=True)
class DeploymentConfiguration:
    memory: int
    concurrency: int


@dataclass(frozen=True)
class TrainingConfiguration:
    model_class: Model
    entry_point: str
    source_dir: str
    instance_type: str
    py_version: str
    framework_version: str


@dataclass(frozen=True)
class PretrainedConfiguration:
    model_class: Model
    model_data: str
    entry_point: str
    source_dir: str
    framework_version: str


class PicklePredictor(Predictor):
    def __init__(self, endpoint_name, sagemaker_session):
        super(PicklePredictor, self).__init__(
            endpoint_name, sagemaker_session, content_type="application/python-pickle"
        )


class ModelHandler:
    _role_handler = RoleHandler()

    @staticmethod
    def create(role_config: SagemakerRoleConfig) -> ModelHandler:
        model_handler = ModelHandler()
        model_handler.initialize(role_config)
        return model_handler

    def initialize(self, role_config: SagemakerRoleConfig) -> None:
        self._role = self._role_handler.initialize_role(role_config)

    def _deploy_model(
        self, model: Model, deployment_config: DeploymentConfiguration
    ) -> Predictor:

        # deployment configuration
        serverless_config = ServerlessInferenceConfig(
            memory_size_in_mb=deployment_config.memory,
            max_concurrency=deployment_config.concurrency,
        )
        predictor = model.deploy(
            serverless_inference_config=serverless_config,
            serializer=IdentitySerializer(),
            deserializer=BytesDeserializer(),
        )
        return predictor

    def train_and_deploy(
        self,
        training_config: TrainingConfiguration,
        deployment_config: DeploymentConfiguration,
    ) -> Predictor:

        model = training_config.model_class(
            entry_point=training_config.entry_point,
            role=self._role.arn,
            source_dir=training_config.source_dir,
            instance_type=training_config.instance_type,
            py_version=training_config.py_version,
            framework_version=training_config.framework_version,
            predictor_cls=PicklePredictor,
        )

        model.fit()
        predictor = self._deploy_model(model, deployment_config)
        return predictor

    def deploy_pretrained(
        self,
        pretrained_config: PretrainedConfiguration,
        deployment_config: DeploymentConfiguration,
    ) -> Predictor:

        model = pretrained_config.model_class(
            model_data=pretrained_config.model_data,
            role=self._role.arn,
            entry_point=pretrained_config.entry_point,
            source_dir=pretrained_config.source_dir,
            framework_version=pretrained_config.framework_version,
        )
        predictor = self._deploy_model(model, deployment_config)
        return predictor

    def tear_down(self, predictor: Predictor) -> None:
        predictor.delete_model()
        predictor.delete_endpoint()

    def predict(self, predictor: Predictor, data: numpy.ndarray) -> numpy.ndarray:
        logger.info(f"Predicting input: {data}")
        payload = pickle.dumps(data)  # nosec
        response = predictor.predict(payload)
        predictions = pickle.loads(response)  # nosec
        logger.info(f"Prediction completed successfully: {predictions}")
        return predictions
