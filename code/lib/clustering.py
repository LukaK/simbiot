#!/usr/bin/env python
import pathlib
from sagemaker.sklearn import SKLearn
from sagemaker.serializers import IdentitySerializer
from sagemaker.deserializers import BytesDeserializer
from sagemaker.serverless import ServerlessInferenceConfig
from .model import MyModel, PicklePredictor


# TODO: Add logging
# TODO: Add support for avilable pretrained models
class ClusteringModel(MyModel):

    # constants
    parrent_path = pathlib.Path(__file__).parent
    clustering_path = parrent_path / "clustering_model"

    def __init__(self, serverless_mode: bool = False):
        super().__init__()

    def _deploy_model(self):
        print("Deploying clustering model...")

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

    # TODO: role not working
    def _train_model(self):
        print("Initializing clustering model instance...")
        self.model = SKLearn(
            "train_and_deploy.py",
            role=self.role_arn,
            instance_type="ml.m5.large",
            source_dir=str(self.clustering_path),
            py_version="py3",
            framework_version="0.23-1",
            predictor_cls=PicklePredictor,
        )
        self.model.fit()
