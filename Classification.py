import numpy as np
import pickle
from Transformations import *
from datetime import datetime
from scipy.stats import mode
import csv

#Number of test data to scale. Full = 10000
datasize = 1000


#Unpickling a file and returning its content
def unpickle(file):
	try:
		import cPickle as pickle
	except:
		import pickle
	fo = open(file, 'rb')
	try:
		dict = pickle.load(fo, encoding='latin1')
	except:
		dict = pickle.load(fo)
	fo.close()
	return dict

""" Calculate the distance between the two images
	Can calculate between matrix v. matrix and matrix vs. vector
	Params: the two images img1 and img2
	Return: the distance between two images"""
def distance(img1,img2):
	#Using Squared Euclidean distance. TODO: find other more effective measures.
	distance = np.sum((img1-img2)**2, axis = 1)
	return distance

""" Normalise the data for better comparison between images
	Params: data z, new_min, new_max
	Return: z normalised"""
#TODO: find the reason why using 0-1000000 is better than 0.0 - 1.0
def normalise(z, new_min=0, new_max=1000000):
	z_min = np.min(z)
	z_max = np.max(z)

	z_normalized = new_min + (z - z_min)*(new_max - new_min)/(z_max - z_min)
	return z_normalized

class kNearestNeighbor:
	def _init_(self):
		pass


	""" Receive training data
		Params: X, a NxD matrix where each row is a training image. N is the data size and D is the image size.
				y, a 1-D array of size N."""
	def train(self, X,y):
		self.Xtr = X
		self.ytr = np.array(y)


	""" Predict the label for the test image
		Params: X, NxD where each row is a test image. N is the data size.
				k, the number of neighbors to consider.
		Return: the label for each test image"""
	def predict(self, X, num_class, k=100):
		num_test = X.shape[0]
		Ypred = np.zeros(num_test, dtype=self.ytr.dtype)

		#loop over all test rows
		for i in range(num_test):
		# find the k nearest training images to the ith test image
			dist = distance(self.Xtr, X[i,:])
			dist += 0.01 #prevent divide by zero

			closest_neighbors = dist.argsort()[:k]
			closest_neighbors_label = []
			for neighbor in closest_neighbors:

				closest_neighbors_label.append(self.ytr[neighbor])

			#count for each label
			num = np.zeros(num_class)

			#for each neighbor, their label has a weight of 1/distance^2
			for j in range(k):
				num[closest_neighbors_label[j]] += 1/(dist[closest_neighbors[j]]**2)
			Ypred[i] = num.argmax()

		return Ypred

class PCA:
    def __init__(self, eigen_vectors, mean, n):
        self.mean = mean
        self.eigen_vectors = eigen_vectors
        self.__n = n
        self.feature_vector = self.eigen_vectors[:n]

    def reduce(self, vector):
        return self.feature_vector.dot(vector - self.mean)

    def reconstruct(self, reduced_vector):
        return reduced_vector.dot(self.feature_vector) + self.mean

    """ Load the training data so to reduce both its and test data's dimensionality
    	Params: data, the training data
    			n, the number of dimensions to be reduced to
    	Return: load the class of eigen_vectors, mean and n i.e.: call init with those values"""
    @classmethod
    def loadData(cls, data, n):
    	#cls is the equivalent of self to normal method
        mean = np.mean(data, axis=0)
        norm_data = data - mean
        #U contains the eigenvalues and V contains the eigenvector
        U, V = np.linalg.eigh(norm_data.T.dot(norm_data))
        V = V.T
        return cls(np.take(V, U.argsort()[::-1], axis=0), mean, n)

if __name__ == "__main__":
	#Define file paths
	meta_file = 'data/batches.meta'
	training_files = ['data/data_batch_1','data/data_batch_2','data/data_batch_3','data/data_batch_4','data/data_batch_5']
	test_file = 'data/test_batch'

	#Obtaining label names
	meta = unpickle(meta_file)
	label_names = meta['label_names']
	num_class = len(label_names)

	#Obtaining training data and labels
	training_data = []			
	training_labels = []

	for f in training_files:
		batch = unpickle(f)
		training_data.extend(batch['data'])
		training_labels.extend(batch['labels'])

	training_data = np.array(training_data)
	training_labels = np.array(training_labels)


	#Obtaining test data
	test = unpickle(test_file)
	test_label = test['labels']
	test_data = test['data']

	""" Run prediction and measure time taken"""
	""" Begin """


	time_start = datetime.now()

	#Normalising both training data and test data
	normalised_training_data = np.apply_along_axis(normalise,1,training_data)
	normalised_test_data = np.apply_along_axis(normalise,1,test_data[0:datasize])

	print("Data normalised")

	#Using PCA to reduce the dimensionality of the data
	pca = PCA.loadData(normalised_training_data, 20)
	reduced_training_data = np.apply_along_axis(pca.reduce, 1, normalised_training_data)
	reduced_test_data = np.apply_along_axis(pca.reduce, 1, normalised_test_data)

	print("Dimensionality reduced")

	#Classifying using nearest neighbor
	kNN = kNearestNeighbor();
	kNN.train(reduced_training_data, training_labels)
	result =  kNN.predict(reduced_test_data, num_class) 

	print("Finished")

	""" Finish  """
	time_finished = datetime.now()
	duration = time_finished - time_start
	print("Time = "+ str(duration))

	#Write to file
	with open('output/output.csv', 'w') as csvfile:
	    writer = csv.writer(csvfile, delimiter=' ',
	                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
	    writer.writerow(result)
