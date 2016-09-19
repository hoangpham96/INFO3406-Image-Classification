import csv
import numpy as np
from scipy.stats import mode
from Classification import datasize, unpickle

if __name__ == "__main__":

	"""Analyze part 1"""
	print("Part 1:")

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

	print("-------------")

	"""Analyze part 2"""
	print("Part 2:")

	with open('output2/output.csv', 'r') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
		try:
			result = list(map(float,reader.__next__()))
			result2 = list(map(float,reader.__next__()))
		except:
			result = map(float,reader.next())
			result2 = map(float,reader.next())

	test2 = unpickle('data2/test')
	test_class_label = test2['fine_labels']
	test_superclass_label = test2['coarse_labels']

	#Calculate the accuracy of the predictions comparing to the lable of the test image
	count = 0.0
	for j in range(datasize):
	    if result[j] == test_class_label[j]:
	        count += 1
	print("Accuracy = {}%".format(count*100/datasize))

	#Calculate the accuracy of the predictions comparing to the lable of the test image
	count = 0.0
	for j in range(datasize):
	    if result2[j] == test_superclass_label[j]:
	        count += 1
	print("Accuracy = {}%".format(count*100/datasize))
