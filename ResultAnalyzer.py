import csv
import numpy as np
from Definitions import datasize, unpickle
import matplotlib.pyplot as plt

if __name__ == "__main__":

	"""Analyze part 1"""
	print("Part 1:")
	print()

	#Read result
	with open('output/output.csv', 'r') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
		try:
			result = list(map(float,reader.__next__()))
		except:
			result = map(float,reader.next())

	result = [int(i) for i in result]

	test = unpickle('data/test_batch')
	test_label = test['labels']

	meta = unpickle('data/batches.meta')
	label_names = meta['label_names']


	#Calculate the accuracy of the predictions comparing to the lable of the test image
	count = 0.0
	for j in range(datasize):
	    if result[j] == test_label[j]:
	        count += 1
	accuracy = count/datasize
	print("Accuracy = {}%".format(accuracy*100))

	#Calculate confusion matrix, precision and recall
	confusion_matrix = np.zeros((len(label_names),len(label_names)))
	for category, hypothesis in zip (test_label,result):
		confusion_matrix[category,hypothesis] += 1

	precision = np.zeros(len(label_names))
	for i in range(len(label_names)):
		precision[i] = confusion_matrix[i][i] / np.sum(confusion_matrix, axis=0)[i]
	print("Precision: {}".format(precision))

	recall = np.zeros(len(label_names))
	for i in range(len(label_names)):
		recall[i] = confusion_matrix[i][i] / np.sum(confusion_matrix, axis=1)[i]
	print("Recall: {}".format(recall))


	"""Plotting confusion matrix"""
	#http://stackoverflow.com/questions/5821125/how-to-plot-confusion-matrix-with-string-axis-rather-than-integer-in-python

	norm_conf = []
	for i in confusion_matrix:
	    a = 0
	    tmp_arr = []
	    a = sum(i, 0)
	    for j in i:
	        tmp_arr.append(float(j)/float(a))
	    norm_conf.append(tmp_arr)

	fig = plt.figure()
	plt.clf()
	ax = fig.add_subplot(1,1,1)
	ax.set_aspect(1)
	res = ax.imshow(np.array(norm_conf), cmap=plt.cm.YlOrRd,
	                interpolation='nearest')

	width, height = confusion_matrix.shape

	for x in range(width):
	    for y in range(height):
	        ax.annotate(str(int(confusion_matrix[x][y])), xy=(y, x),
	                    horizontalalignment='center',
	                    verticalalignment='center')

	cb = fig.colorbar(res)
	plt.xticks(range(width), label_names[:width], rotation = -45)
	plt.yticks(range(height), label_names[:height])

	plt.savefig('plots/confusion_matrix_part1.png', format='png')
	plt.close()

	print()
	print("---------------")
	print()





	"""Analyze part 2"""
	print("Part 2:")
	print()

	#Read result
	meta = unpickle('data2/meta')
	class_names = meta['fine_label_names']
	superclass_names = meta['coarse_label_names']

	with open('output2/output.csv', 'r') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
		try:
			result = list(map(float,reader.__next__()))
			#Fixing a bug where the writer would put extra line in between the two output
			temp = list(map(float,reader.__next__()))
			if len(temp) != 0:
				result2 = temp
			else:
				result2 = list(map(float,reader.__next__()))
		except:
			result = map(float,reader.next())
			#Fixing a bug where the writer would put extra line in between the two output
			temp = map(float,reader.next())
			if len(temp) != 0:
				result2 = temp
			else:
				result2 = map(float,reader.next())

	test2 = unpickle('data2/test')
	test_class_label = test2['fine_labels']
	test_superclass_label = test2['coarse_labels']

	result = [int(i) for i in result]
	result2 = [int(i) for i in result2]






	print("Fine labels:")
	#Calculate the accuracy of the predictions comparing to the lable of the test image
	count = 0.0
	for j in range(datasize):
	    if result[j] == test_class_label[j]:
	        count += 1
	print("Accuracy = {}%".format(count*100/datasize))


	#Calculate confusion matrix, precision and recall
	confusion_matrix = np.zeros((len(class_names),len(class_names)))
	for category, hypothesis in zip (test_class_label,result):
		confusion_matrix[category,hypothesis] += 1

	precision = np.zeros(len(class_names))
	for i in range(len(class_names)):
		precision[i] = confusion_matrix[i][i] / np.sum(confusion_matrix, axis=0)[i]
	print("Precision: {}".format(precision))

	recall = np.zeros(len(class_names))
	for i in range(len(class_names)):
		recall[i] = confusion_matrix[i][i] / np.sum(confusion_matrix, axis=1)[i]
	print("Recall: {}".format(recall))


	"""Plotting confusion matrix"""
	#http://stackoverflow.com/questions/5821125/how-to-plot-confusion-matrix-with-string-axis-rather-than-integer-in-python

	norm_conf = []
	for i in confusion_matrix:
	    a = 0
	    tmp_arr = []
	    a = sum(i, 0)
	    for j in i:
	        tmp_arr.append(float(j)/float(a))
	    norm_conf.append(tmp_arr)

	fig = plt.figure()
	plt.clf()
	ax = fig.add_subplot(1,1,1)
	ax.set_aspect(1)
	res = ax.imshow(np.array(norm_conf), cmap=plt.cm.YlOrRd,
	                interpolation='nearest')

	width, height = confusion_matrix.shape

	cb = fig.colorbar(res)

	plt.savefig('plots/confusion_matrix_part2_fine.png', format='png')
	plt.close()





	print()
	print("***")
	print()
	print("Coarse labels:")

	#Calculate the accuracy of the predictions comparing to the lable of the test image
	count = 0.0
	for j in range(datasize):
	    if result2[j] == test_superclass_label[j]:
	        count += 1
	print("Accuracy = {}%".format(count*100/datasize))


	#Calculate confusion matrix, precision and recall
	confusion_matrix = np.zeros((len(superclass_names),len(superclass_names)))
	for category, hypothesis in zip (test_superclass_label,result2):
		confusion_matrix[category,hypothesis] += 1

	precision = np.zeros(len(superclass_names))
	for i in range(len(superclass_names)):
		precision[i] = confusion_matrix[i][i] / np.sum(confusion_matrix, axis=0)[i]
	print("Precision: {}".format(precision))

	recall = np.zeros(len(superclass_names))
	for i in range(len(superclass_names)):
		recall[i] = confusion_matrix[i][i] / np.sum(confusion_matrix, axis=1)[i]
	print("Recall: {}".format(recall))



	"""Plotting confusion matrix"""
	#http://stackoverflow.com/questions/5821125/how-to-plot-confusion-matrix-with-string-axis-rather-than-integer-in-python

	norm_conf = []
	for i in confusion_matrix:
	    a = 0
	    tmp_arr = []
	    a = sum(i, 0)
	    for j in i:
	        tmp_arr.append(float(j)/float(a))
	    norm_conf.append(tmp_arr)

	fig = plt.figure()
	plt.clf()
	ax = fig.add_subplot(1,1,1)
	ax.set_aspect(1)
	res = ax.imshow(np.array(norm_conf), cmap=plt.cm.YlOrRd,
	                interpolation='nearest')

	width, height = confusion_matrix.shape

	cb = fig.colorbar(res)

	plt.xticks(range(width), range(1,width+1))
	plt.yticks(range(height), range(1,height+1))

	plt.savefig('plots/confusion_matrix_part2_coarse.png', format='png')
	plt.close()
