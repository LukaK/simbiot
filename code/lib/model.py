#!/usr/bin/env python
import numpy
import pickle  # nosec
from .logger import logger
from dataclasses import dataclass
from sagemaker.predictor import Predictor
from sagemaker.serializers import IdentitySerializer
from sagemaker.deserializers import BytesDeserializer
from sagemaker.serverless import ServerlessInferenceConfig
from .role import RoleHandler, SagemakerRoleConfig

# TODO: change object to Model class


@dataclass(frozen=True)
class DeployedModel:
    model: object
    predictor: Predictor


@dataclass(frozen=True)
class DeploymentConfiguration:
    memory: int
    concurrency: int


@dataclass(frozen=True)
class TrainingConfiguration:
    entry_point: str
    source_dir: str
    instance_type: str
    py_version: str
    framework_version: str


@dataclass(frozen=True)
class PretrainedConfiguration:
    model_data: str
    entry_point: str
    source_dir: str
    framework_version: str


# TODO: Add check if at least one of the deployment methods is defined
@dataclass(frozen=True)
class ModelConfiguration:
    deployment: DeploymentConfiguration
    training: TrainingConfiguration
    pretrained: PretrainedConfiguration


class PicklePredictor(Predictor):
    def __init__(self, endpoint_name, sagemaker_session):
        super(PicklePredictor, self).__init__(
            endpoint_name, sagemaker_session, content_type="application/python-pickle"
        )


class ModelHandler:
    _logger = logger
    _role_handler = RoleHandler()

    def initialize(self, role_name: str = "MySagemakerRole"):
        self._role = self._role_handler.initialize_role(
            SagemakerRoleConfig(name=role_name)
        )

    def _deploy_model(
        self, model: object, model_configuration: ModelConfiguration
    ) -> Predictor:

        # deployment configuration
        serverless_config = ServerlessInferenceConfig(
            memory_size_in_mb=model_configuration.deployment.memory,
            max_concurrency=model_configuration.deployment.concurrency,
        )
        predictor = model.deploy(
            serverless_inference_config=serverless_config,
            serializer=IdentitySerializer(),
            deserializer=BytesDeserializer(),
        )
        return predictor

    def train_and_deploy(
        self, model_class: object, model_configuration: ModelConfiguration
    ) -> DeployedModel:
        self._logger.info(
            f"Training/deploying clustering model with configuration: {model_configuration}"
        )
        model = model_class(
            entry_point=model_configuration.training.entry_point,
            role=self._role.arn,
            source_dir=model_configuration.training.source_dir,
            instance_type=model_configuration.training.instance_type,
            py_version=model_configuration.training.py_version,
            framework_version=model_configuration.training.framework_version,
            predictor_cls=PicklePredictor,
        )

        model.fit()
        predictor = self._deploy_model(model, model_configuration)
        return DeployedModel(model=model, predictor=predictor)

    def deploy_pretrained(
        self, model_class: object, model_configuration: ModelConfiguration
    ) -> DeployedModel:

        self._logger.info(
            f"Configuring pretrained clustering model: {model_configuration}"
        )
        model = model_class(
            model_data=model_configuration.pretrained.model_location,
            role=self._role.arn,
            entry_point=model_configuration.pretrained.entry_point,
            source_dir=model_configuration.pretrained.source_dir,
            framework_version=model_configuration.pretrained.framework_version,
        )
        predictor = self._deploy_model(model, model_configuration)
        return DeployedModel(model=model, predictor=predictor)

    def tear_down(self, deployed_model: DeployedModel) -> None:
        self._logger.info("Tearing down deployed model")
        deployed_model.predictor.delete_model()
        deployed_model.predictor.delete_endpoint()
        self._logger.info("Teardown completed successfully")

    def predict(
        self, deployed_model: DeployedModel, data: numpy.ndarray
    ) -> numpy.ndarray:
        self._logger.info(f"Predicting input: {data}")
        payload = pickle.dumps(data)  # nosec
        response = deployed_model.predictor.predict(payload)
        predictions = pickle.loads(response)  # nosec
        self._logger.info(f"Prediction completed successfully: {predictions}")
        return predictions
