import numpy as np
import pickle
from Transformations import *
from datetime import datetime
from scipy.stats import mode
import csv

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

#Define file paths
meta_file = 'data/batches.meta'
training_files = ['data/data_batch_1','data/data_batch_2','data/data_batch_3','data/data_batch_4','data/data_batch_5']
test_file = 'data/test_batch'

#Obtaining label names
meta = unpickle(meta_file)
label_names = meta['label_names']

#Obtaining training data and lables
training_data = []					#Array of 5 training batches, each batches contain 1000 img
training_lables = []

for f in training_files:
	batch = unpickle(f)
	training_data.append(batch['data'])
	training_lables.append(batch['labels'])

#Obtaining test data
test = unpickle(test_file)
test_label = test['labels']
test_data = test['data']

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


	""" Predict the lable for the test image
		Params: X, NxD where each row is a test image. N is the data size.
				k, the number of neighbors to consider.
		Return: the lable for each test image"""
	def predict(self, X, k=100):
		num_test = X.shape[0]
		Ypred = np.zeros(num_test, dtype=self.ytr.dtype)

		#loop over all test rows
		for i in range(num_test):
		# find the k nearest training images to the ith test image
			dist = distance(self.Xtr, X[i,:])

			closest_neighbors = dist.argsort()[:k]
			closest_neighbors_lable = []
			for neighbor in closest_neighbors:
				closest_neighbors_lable.append(self.ytr[neighbor])

			#count for each lable
			num = np.zeros(10)

			#for each neighbor, their lable has a weight of 1/distance^2
			for j in range(k):
				num[closest_neighbors_lable[j]] += 1/(dist[closest_neighbors[j]]**2)

			Ypred[i] = num.argmax()

		return Ypred

if __name__ == "__main__":
	""" Test accuracy and measure time taken"""
	""" Begin testing """

	result = []

	for batch_num in range(5):
		time_start = datetime.now()

		#Normalising both training data and test data
		normalised_training_data = []
		normalised_test_data = []
		for i in range(datasize):
			normalised_training_data.append( normalise(training_data[batch_num][i]) )
			normalised_test_data.append( normalise(test_data[i]) )
		normalised_training_data = np.array(normalised_training_data)
		normalised_test_data = np.array( normalised_test_data )

		#Classifying using nearest neighbor
		kNN = kNearestNeighbor();
		kNN.train(normalised_training_data,training_lables[batch_num][0:datasize])
		result.append( kNN.predict(normalised_test_data) )

		print("Batch {} done!".format(batch_num+1))

		""" Finish testing """
		time_finished = datetime.now()
		duration = time_finished - time_start
		print("Time = "+ str(duration))


	""" Find the best result.
		The lable with the most number of predictions is the best result"""
	result_matrix = np.matrix(result).T
	best_result = np.zeros(result_matrix.shape[0])
	for i in range(result_matrix.shape[0]):
	    best_est = np.squeeze(np.asarray(result_matrix[i]))
	    best_result[i] = int(mode(best_est).mode[0])

	#Write to file
	with open('output/output.csv', 'w') as csvfile:
	    writer = csv.writer(csvfile, delimiter=' ',
	                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
	    writer.writerow(best_result)


# """ Test transformation functions by visualizing the images"""
# pylab.figure()
# pylab.gray()
# pylab.imshow(rgb2gray(mirror(training_data[0][100],False)).reshape(32,32))

# pylab.figure()
# pylab.gray()
# pylab.imshow(rgb2gray(training_data[0][100]).reshape(32,32))

# pylab.show()
