#!/usr/bin/env python
from sagemaker.deserializers import BytesDeserializer
from sagemaker.serializers import IdentitySerializer
from sagemaker.serverless import ServerlessInferenceConfig
from sagemaker.sklearn import SKLearn, SKLearnModel

from .model import ModelWrapper, PicklePredictor


class Clustering(ModelWrapper):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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

    def _use_pretrained_model(self):
        self.logger.info("Configuring pretrained clustering model")
        self.model = SKLearnModel(
            model_data=self.model_location,
            role=self.role_arn,
            entry_point="clustering.py",
            source_dir=str(self.entry_path),
            framework_version="0.23-1",
        )

    def _train_model(self):
        self.logger.info("Training clustering model")
        self.model = SKLearn(
            entry_point="clustering.py",
            role=self.role_arn,
            source_dir=str(self.entry_path),
            instance_type="ml.m5.large",
            py_version="py3",
            framework_version="0.23-1",
            predictor_cls=PicklePredictor,
        )

        self.model.fit()
        self.logger.info("Model training completed successfully")
