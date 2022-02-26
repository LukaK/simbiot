#!/usr/bin/env python
import pickle  # nosec
import numpy
import logging
from sklearn.cluster import DBSCAN

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def model_fn(model_dir: str) -> DBSCAN:
    logger.info("Loading model")
    model = DBSCAN(eps=10, min_samples=2)
    return model


def input_fn(
    serialized_input_data: bytes, content_type: str = "application/python-pickle"
) -> numpy.ndarray:
    print(f"Content type: {content_type}")
    logger.info(f"Deserializing the input data: {serialized_input_data}")
    try:
        data = pickle.loads(serialized_input_data)  # nosec
        return data
    except:
        raise Exception(
            f"Requested unsupported ContentType in content_type: {content_type}"
        )


def predict_fn(input_object: numpy.ndarray, model: DBSCAN) -> numpy.ndarray:
    logger.info("Predict function...")
    logger.info(f"Input object: {input_object}")
    results = model.fit_predict(input_object)
    logger.info(f"Predictions: {results}")
    return results


def output_fn(prediction: numpy.ndarray, content_type: str) -> bytes:
    logger.info("Serializing the generated output.")
    output = pickle.dumps(prediction)
    return output
