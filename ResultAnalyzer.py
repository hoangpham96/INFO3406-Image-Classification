import csv
import numpy as np
from scipy.stats import mode

datasize = 1000

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


for batch_num in range(5):
    with open('output/output{}.csv'.format(batch_num+1), 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        result.append( map(int,reader.next()) )


test = unpickle('data/test_batch')
test_label = test['labels']

result_matrix = np.matrix(result).T
best = np.zeros(result_matrix.shape[0])
for i in range(result_matrix.shape[0]):
    best_ests = np.squeeze(np.asarray(result_matrix[i]))
    best[i] = mode(best_ests).mode[0]


for i in range(5):
    count = 0.0
    for j in range(datasize):
        if result[i][j] == test_label[j]:
            count += 1
    print("Accuracy = {}%".format(count*100/datasize))

print('---------------')
count = 0.0
for j in range(datasize):
    if best[j] == test_label[j]:
        count += 1
print("Accuracy = {}%".format(count*100/datasize))
