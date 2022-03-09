#!/usr/bin/env python
import os
import pickle  # nosec
import argparse
import numpy
from sklearn.cluster import DBSCAN
import joblib


def model_fn(model_dir: str) -> DBSCAN:
    model = joblib.load(os.path.join(model_dir, "clustering.joblib"))
    return model


def input_fn(
    serialized_input_data: bytes, content_type: str = "application/python-pickle"
) -> numpy.ndarray:
    data = pickle.loads(serialized_input_data)  # nosec
    return data


def predict_fn(input_object: numpy.ndarray, model: DBSCAN) -> numpy.ndarray:
    results = model.fit_predict(input_object)
    return results


def output_fn(prediction: numpy.ndarray, content_type: str) -> bytes:
    output = pickle.dumps(prediction)
    return output


if __name__ == "__main__":

    # get output directory
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    args, _ = parser.parse_known_args()

    # save clustering model to the s3 bucket
    model = DBSCAN(eps=10, min_samples=2)
    joblib.dump(model, os.path.join(args.model_dir, "clustering.joblib"))
