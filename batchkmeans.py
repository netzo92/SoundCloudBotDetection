#Written by Metehan Ozten 
# use cmdline arg '-i' to specify input file
# use cmdline arg '-v' to turn on verbose printing

import matplotlib.pyplot as plt
import numpy
import sys
from sklearn.cluster import MiniBatchKMeans, KMeans

def print_centroids(clusters):
	print('Data Columns: public_favorites_count|followings_count|followers_count|likes_count|track_count|playlist_count|comments_count')
	for x in range(0, clusters.shape[0]):
		print('Centroid %d: %s' % (x, str(clusters[x])))


def print_data_with_labels(data, analysis_data, labels, spammy_ids, num_clusters):
	k = [0 for x in range(0, num_clusters)]
	for i in range(0, data.shape[0]):
		if int(analysis_data[i][0]) not in spammy_ids:
			print('user_id: %d id: %d %s label: %d' % (analysis_data[i][0], analysis_data[i][1], str(data[i]), labels[i]))
		else:
			print('spammer_user_id: %d id: %d %s label: %d' % (analysis_data[i][0], analysis_data[i][1], str(data[i]), labels[i]))
			k[labels[i]] += 1
	print(k)
	return k

def read_spam_ids(filename = 'spam_only_ids.csv'):
	spammy_ids = {}
	with open(filename, 'r') as f:
		for line in f:
			num = int(line)
			spammy_ids[num] = True
	return spammy_ids


def main():
	use_cols = (2,3,4,5,6,7,8)
	if '-i' not in sys.argv:
		analysis_data = numpy.loadtxt('features.txt', skiprows = 1, delimiter = ',')
		data = numpy.loadtxt('features.txt', skiprows = 1, usecols = use_cols, delimiter = ',')
	else :
		analysis_data = numpy.loadtxt(sys.argv[sys.argv.index('-i')+1], skiprows = 1, delimiter = ',')
		data = numpy.loadtxt(sys.argv[sys.argv.index('-i')+1], skiprows = 1, usecols = use_cols, delimiter = ',')
	num_clusters = 4 #default
	if '-n' in sys.argv:
		num_clusters = int(sys.argv[sys.argv.index('-n')+1])
	our_model = MiniBatchKMeans(n_clusters = num_clusters, batch_size = 2000)
	labels = our_model.fit_predict(data) 
	clusters = our_model.cluster_centers_
	spammy_ids = read_spam_ids()
	title = "number of clusters: %d" % (len(clusters))
	print_centroids(clusters)
	if '-v' in sys.argv:
		print_data_with_labels(data, analysis_data, labels, spammy_ids, num_clusters)
	
if __name__ == '__main__':
	main()

