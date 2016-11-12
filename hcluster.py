import matplotlib.pyplot as plt
import numpy
import scipy.cluster.hierarchy as hcluster

data = numpy.loadtxt('features20000.txt', skiprows = 1, delimiter = ',')

# clustering
thresh = 1.5
clusters = hcluster.fclusterdata(data, thresh, criterion="distance")

# plotting
#plt.scatter(*numpy.transpose(data), c=clusters)
#plt.axis("equal")
title = "threshold: %f, number of clusters: %d" % (thresh, len(set(clusters)))
plt.title(title)
plt.show()
