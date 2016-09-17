import numpy as np
import matplotlib as pl
import pickle
import pylab
from Transformations import *
from datetime import datetime
from scipy.stats import mode
import csv

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

def distance(img1,img2):
	#Using Euclidean distance. TODO: find other more effective measures.
	distance = np.sqrt(np.sum((img1-img2)**2, axis = 1))
	return distance

def normalise(z, new_min=1.0, new_max=2.0):
	#function definition
	z_min = np.min(z)
	z_max = np.max(z)

	z_normalized = new_min + (z - z_min)*(new_max - new_min)/(z_max - z_min)
	return z_normalized

class kNearestNeighbor:
	def _init_(self):
		pass

	def train(self, X,y):
		"""input	X -> training set features
					y -> labels of training sets"""
		""" X is N x D where each row is a training image. Y is 1-dimension of size N """
		# the nearest neighbor classifier simply remembers all the training data
		self.Xtr = X
		self.ytr = np.array(y)

	def predict(self, X, k):
		""" X is N x D where each row is a test image we wish to predict label for """
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

			num = np.zeros(10)

			for j in range(k):
				num[closest_neighbors_lable[j]] += 1/(dist[closest_neighbors[j]]**2)

			# min_index = np.argmin(distance(self.Xtr, X[i,:])) #get the index with smallest distance
			# Ypred[i] = mode(closest_neighbors_lable).mode[0] #predict the label of the nearest example
			Ypred[i] = num.argmax()

		return Ypred


""" Test accuracy and measure time taken"""
""" Begin testing """
time_start = datetime.now()

datasize = 1000

a = []
b = []
for i in range(datasize):
	a.append( normalise(training_data[0][i]))
	b.append( normalise(test_data[i]))
a = np.array(a)
b = np.array(b)


kNN = kNearestNeighbor();
kNN.train(a,training_lables[0][0:datasize])
result = kNN.predict(b, 100)

with open('output/output.csv', 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(result)

count = 0
for i in range(datasize):
	 if result[i] == test_label[i]:
	 	count += 1
print("Accuracy = " + str(count*100/datasize) + "%")

time_finished = datetime.now()

""" Finish testing """

duration = time_finished - time_start
print("Time = "+ str(duration))



# #Red channel of a picture with lable 1
# similar_pictures = []
# for i in range(0,len(training_data[0])):
# 	if training_lables[0][i] == 1:
# 		pic_red_channel = training_data[0][i][0:1024]
# 		similar_pictures.append(pic_red_channel)
#
# #Red channel of a picture with different lable
# similar_pictures2 = []
# for i in range(0,len(training_data[0])):
# 	if training_lables[0][i] == 6:
# 		pic_red_channel = training_data[0][i][0:1024]
# 		similar_pictures2.append(pic_red_channel)
#
# # print(distance(similar_pictures[0],similar_pictures[10]))




# """ Test functions by visualizing the images"""
# pylab.figure()
# pylab.gray()
# pylab.imshow(rgb2gray(mirror(training_data[0][100],False)).reshape(32,32))

# pylab.figure()
# pylab.gray()
# pylab.imshow(rgb2gray(training_data[0][100]).reshape(32,32))

# pylab.show()
