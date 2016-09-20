from Transformations import * #unused
from Definitions import *

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
