#Written by Metehan Ozten 
# use cmdline arg '-i' to specify input file
# use cmdline arg '-v' to turn on verbose printing
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy
import sys
from sklearn.cluster import MiniBatchKMeans, KMeans
VERBOSE = True if '-v' in sys.argv else False
FEATURES = 'Data Columns: 1-Followers, 2-Following,3-Published,4-URL,5-ProfaneDescription,6-ProfaneTitle,7-Action,8-Pays,9-DuplicateDescription,10-DuplicateWebsite,11-website_aggregate,12-Spam' 
FEAT_LIST = ['Followers', 'Following','Published', 'URL', 'ProfaneDesc', 'ProfaneTitle', 'Action', 'Pays', 'DuplicateDescription', 'DuplicateWebsite', 'website']

def print_centroids(clusters):
	print(FEATURES)
	for x in range(0, clusters.shape[0]):
		print('Centroid %d: %s' % (x, str(clusters[x])))

def vec_to_string(vec):
	astr = ''
	for x in range(0,vec.shape[0]):
		astr += '%d ,' % vec[x]
	return astr

	
		

def print_data_with_labels(data, analysis_data, labels, spammy_ids, num_clusters):
	k = [0 for x in range(0, num_clusters)]
	centroid_buckets = [0 for x in range(0, num_clusters)]
	for i in range(0, data.shape[0]):
		centroid_buckets[labels[i]] += 1
		if int(analysis_data[i][0]) not in spammy_ids:
			if VERBOSE:
				print('user_id: %d %s label: %d' % (analysis_data[i][0], vec_to_string(data[i]), labels[i]))
		else:
			if VERBOSE:
				print('spammer_user_id: %d %s label: %d' % (analysis_data[i][0],  vec_to_string(data[i]), labels[i]))
			k[labels[i]] += 1
	print(k)
	print('Centroid Buckets: '+str(centroid_buckets))
	return k

def read_spam_ids(filename = 'spam_only_ids.csv'):
	spammy_ids = {}
	with open(filename, 'r') as f:
		for line in f:
			num = int(line)
			spammy_ids[num] = True
	return spammy_ids


def main():
	use_cols = (1,2,3,4,5,6,7,8,9,10,11)
	if '-i' not in sys.argv:
		analysis_data = numpy.loadtxt('features.txt', skiprows = 1, delimiter = ',')
		data = numpy.loadtxt('features.txt', skiprows = 1, usecols = use_cols, delimiter = ',')
	else :
		analysis_data = numpy.loadtxt(sys.argv[sys.argv.index('-i')+1], skiprows = 1, delimiter = ',')
		data = numpy.loadtxt(sys.argv[sys.argv.index('-i')+1], skiprows = 1, usecols = use_cols, delimiter = ',')
	num_clusters = 4 #default
	if '-n' in sys.argv:
		num_clusters = int(sys.argv[sys.argv.index('-n')+1])
	scaled_data = preprocessing.scale(data)
	our_model = MiniBatchKMeans(n_clusters = num_clusters, batch_size = 2000, n_init = 5)
	labels = our_model.fit_predict(scaled_data) 
	clusters = our_model.cluster_centers_
	inertia = our_model.inertia_
	print(inertia)
	spammy_ids = read_spam_ids()
	title = "number of clusters: %d" % (len(clusters))
	print_centroids(clusters)
	print_data_with_labels(scaled_data, analysis_data, labels, spammy_ids, num_clusters)
	
if __name__ == '__main__':
	main()

