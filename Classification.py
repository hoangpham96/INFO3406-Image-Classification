import numpy as np
import matplotlib as pl
import pickle

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
training_data = []
training_lables = []

for f in training_files:
	batch = unpickle(f)
	training_data.append(batch['data'])
	training_lables.append(batch['labels'])

#Obtaining test data
test = unpickle(test_file)
test_label = test['labels']
test_data = test['data']

def PCA(img):
	