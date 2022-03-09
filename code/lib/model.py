#!/usr/bin/env python
import boto3
import numpy
import pickle  # nosec
import pathlib
import sagemaker
from abc import ABC, abstractmethod
from sagemaker.predictor import Predictor
from .logger import Logger


# TODO: Add role creation if missing
class ModelWrapper(ABC):

    # constants
    role_name = "MySagemakerRole"
    parrent_path = pathlib.Path(__file__).parent
    entry_path = parrent_path / "model_hosting"

    # resources
    sagemaker_session = sagemaker.Session()
    iam_client = boto3.client("iam")
    logger = Logger.get_logger()

    def __init__(self):
        self.model = None
        self.predictor = None
        self.role_arn = self.iam_client.get_role(RoleName=self.role_name)["Role"]["Arn"]

    def initialize(self):
        self.logger.info("Initializing model")
        self._train_model()
        self._deploy_model()
        self.logger.info("Model initialization completed successfully")

    def cleanup(self):
        self.logger.info("Cleaning up resources")
        if self.predictor:
            self.logger.info("Deleting model resource")
            self.predictor.delete_model()

            self.logger.info("Deleting endpoing resource")
            self.predictor.delete_endpoint()
        self.logger.info("Cleanup completed successfully")

    # TODO: Test predictions
    def predict(self, data: numpy.ndarray) -> numpy.ndarray:
        payload = pickle.dumps(data)  # nosec
        response = self.predictor.predict(payload)
        predictions = pickle.loads(response)  # nosec
        return predictions

    @abstractmethod
    def _train_model(self):
        pass

    @abstractmethod
    def _deploy_model(self):
        pass


class PicklePredictor(Predictor):
    def __init__(self, endpoint_name, sagemaker_session):
        super(PicklePredictor, self).__init__(
            endpoint_name, sagemaker_session, content_type="application/python-pickle"
        )
