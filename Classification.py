import numpy as np
import matplotlib as pl
import pickle
import pylab

#Unpickling a file and returning its content
def unpickle(file):
	try:
		import cPickle as pickle
	except:
		import pickle
	fo = open(file, 'rb')
	dict = pickle.load(fo, encoding='latin1')
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
	distance = (img1.astype("float")-img2.astype("float"))**2
	distance = np.sqrt(sum(distance))
	return distance

def rgb2gray(img):
	r, g, b = img[0:1024], img[1024:2048], img[2048:3072]
	gray = 0.2989 * r.astype("float") + 0.5870 * g.astype("float") + 0.1140 * b.astype("float")				#Formula to turn rgb to grayscale
	return gray




# #Below is a series of test
# #
# #Red channel of a picture with lable 1
# similar_pictures = []
# for i in range(0,len(training_data[0])):
# 	if training_lables[0][i] == 1:
# 		pic_red_channel = training_data[0][i][0:1024]
# 		similar_pictures.append(pic_red_channel)

# #Red channel of a picture with different lable 
# similar_pictures2 = []
# for i in range(0,len(training_data[0])):
# 	if training_lables[0][i] == 6:
# 		pic_red_channel = training_data[0][i][0:1024]
# 		similar_pictures2.append(pic_red_channel)



# # print(distance(similar_pictures[0],similar_pictures[10]))

# print(training_lables[0][0])

# pylab.figure()
# pylab.gray()
# pylab.imshow(similar_pictures2[0].reshape(32,32))


# pylab.figure()
# pylab.gray()
# pylab.imshow(rgb2gray(training_data[0][0]).reshape(32,32))

# pylab.show()