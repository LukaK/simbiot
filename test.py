#!/usr/bin/env python
import numpy
from lib import Clustering


clustering_model_location = "s3://sagemaker-us-east-1-399446234556/clustering.tar.gz"

test_data = numpy.array([1, 2, 3, 4, 5, 10, 11, 22])
test_data = test_data.reshape(-1, 1)
clustering = Clustering(model_location=clustering_model_location)
clustering.initialize()
clustering.predict(test_data)
clustering.cleanup()
