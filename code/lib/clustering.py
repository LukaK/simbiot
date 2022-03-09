#!/usr/bin/env python
from sagemaker.sklearn import SKLearn
from sagemaker.serializers import IdentitySerializer
from sagemaker.deserializers import BytesDeserializer
from sagemaker.serverless import ServerlessInferenceConfig
from .model import ModelWrapper, PicklePredictor


# TODO: Add support for avilable pretrained models
class Clustering(ModelWrapper):
    def __init__(self, serverless_mode: bool = False):
        super().__init__()

    def _deploy_model(self):
        self.logger.info("Deploying clustering model")

        # deployment configuration
        serverless_config = ServerlessInferenceConfig(
            memory_size_in_mb=4096,
            max_concurrency=10,
        )
        self.predictor = self.model.deploy(
            serverless_inference_config=serverless_config,
            serializer=IdentitySerializer(),
            deserializer=BytesDeserializer(),
        )
        self.logger.info("Model deployment completed successfully")

    def _train_model(self):
        self.logger.info("Training clustering model")
        self.model = SKLearn(
            "clustering.py",
            role=self.role_arn,
            instance_type="ml.m5.large",
            source_dir=str(self.entry_path),
            py_version="py3",
            framework_version="0.23-1",
            predictor_cls=PicklePredictor,
        )
        self.model.fit()
        self.logger.info("Clustering model trained successfully")
