#!/usr/bin/env python
import numpy
from lib import Clustering

test_data = numpy.array([1, 2, 3, 4, 5, 10, 11, 22])
test_data = test_data.reshape(-1, 1)
clustering = Clustering()
clustering.initialize()
clustering.predict(test_data)
clustering.cleanup()
