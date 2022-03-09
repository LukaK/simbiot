#!/usr/bin/env python
import boto3
import numpy
import pickle  # nosec
import sagemaker
from abc import ABC, abstractmethod
from sagemaker.predictor import Predictor


# TODO: Add logging
# TODO: Add role creation if missing
class MyModel(ABC):

    # constants
    role_name = "MySagemakerRole"

    # resources
    sagemaker_session = sagemaker.Session()
    iam_client = boto3.client("iam")

    def __init__(self):
        self.model = None
        self.predictor = None
        self.role_arn = self.iam_client.get_role(RoleName=self.role_name)["Role"]["Arn"]

    def initialize(self):
        self._train_model()
        self._deploy_model()

    def cleanup(self):
        print("Cleaning up clustering resources...")

        if self.predictor:
            self.predictor.delete_model()
            self.predictor.delete_endpoint()

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
