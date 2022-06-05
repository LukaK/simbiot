#!/usr/bin/env python
import time
import json
import boto3
import numpy
import pickle  # nosec
import pathlib
import sagemaker
from abc import ABC, abstractmethod
from sagemaker.predictor import Predictor
from .logger import Logger


# TODO: Add support without training
class ModelWrapper(ABC):

    # resources
    sagemaker_session = sagemaker.Session()
    iam_client = boto3.client("iam")
    logger = Logger.get_logger()

    # constants
    role_name = "MySagemakerRole"
    parrent_path = pathlib.Path(__file__).parent
    entry_path = parrent_path / "model_hosting"
    trust_policy_document = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "sagemaker.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }

    def __init__(self, model_location: str = None):
        self.model = None
        self.predictor = None
        self.model_location = model_location
        self.role_arn = self._setup_sagemaker_role()

    @classmethod
    def _setup_sagemaker_role(cls) -> str:
        try:
            role_arn = cls.iam_client.get_role(RoleName=cls.role_name)["Role"]["Arn"]
        except Exception:
            cls.logger.info("Creating iam role for the sagemaker.")

            # create role
            role_arn = cls.iam_client.create_role(
                RoleName=cls.role_name,
                AssumeRolePolicyDocument=json.dumps(cls.trust_policy_document),
            )["Role"]["Arn"]

            # attach policy
            cls.iam_client.attach_role_policy(
                RoleName=cls.role_name,
                PolicyArn="arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
            )
            time.sleep(5)
            cls.logger.info("Iam role created successfully.")
        return role_arn

    def initialize(self):
        self.logger.info("Initializing model")

        if self.model_location:
            self._use_pretrained_model()
        else:
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

    def predict(self, data: numpy.ndarray) -> numpy.ndarray:
        """Use model for inference

        Args:
            data (ndarray): mxn array, m datapoints of dimension n

        Returns:
            ndarray: Inference results
        """

        self.logger.info(f"Predicting input: {data}")
        payload = pickle.dumps(data)  # nosec
        response = self.predictor.predict(payload)
        predictions = pickle.loads(response)  # nosec
        self.logger.info(f"Prediction completed successfully: {predictions}")
        return predictions

    @abstractmethod
    def _train_model(self):
        pass

    @abstractmethod
    def _deploy_model(self):
        pass

    @abstractmethod
    def _use_pretrained_model(self):
        pass


class PicklePredictor(Predictor):
    def __init__(self, endpoint_name, sagemaker_session):
        super(PicklePredictor, self).__init__(
            endpoint_name, sagemaker_session, content_type="application/python-pickle"
        )
