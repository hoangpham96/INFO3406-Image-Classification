import csv
import numpy as np
from scipy.stats import mode
from Classification import datasize

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

result = []

#Read result
with open('output/output.csv', 'r') as csvfile:
	reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
	try:
		result = list(map(float,reader.__next__()))
	except:
		result = map(float,reader.next())

test = unpickle('data/test_batch')
test_label = test['labels']

#Calculate the accuracy of the predictions comparing to the lable of the test image
count = 0.0
for j in range(datasize):
    if result[j] == test_label[j]:
        count += 1
print("Accuracy = {}%".format(count*100/datasize))
