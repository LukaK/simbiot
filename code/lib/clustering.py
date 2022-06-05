#!/usr/bin/env python
from sagemaker.sklearn import SKLearn
from sagemaker.serializers import IdentitySerializer
from sagemaker.deserializers import BytesDeserializer
from sagemaker.serverless import ServerlessInferenceConfig
from .model import ModelWrapper, PicklePredictor


# TODO: Test support for pretrained models
# TODO: Add validation for model location
class Clustering(ModelWrapper):
    def __init__(self, model_location: str = None):
        self._model_location = model_location
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

    def _train_or_prepare_model(self):
        self.logger.info("Starting clustering model preparation")
        self.model = SKLearn(
            model_data=self._model_location,
            entry_point="clustering.py",
            role=self.role_arn,
            source_dir=str(self.entry_path),
            instance_type="ml.m5.large",
            py_version="py3",
            framework_version="0.23-1",
            predictor_cls=PicklePredictor,
        )

        # If there is no pretrained model train it
        if not self._model_location:
            self.logger.info("Training clustering model")
            self.model.fit()

        self.logger.info("Model preparation completed successfully")
