from Definitions import *
from Transformations import depixelize
import os
from PIL import Image

if __name__ == "__main__":
	#Define file paths
	meta_file = 'data/batches.meta'
	training_files = ['data/data_batch_1','data/data_batch_2','data/data_batch_3','data/data_batch_4','data/data_batch_5']
	query_folder = 'INFO3406_assignment1_query/'

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

	#Assuming that the query image is the same in format as the png in the example
	#The program will not work otherwise
	query_data = []
	# Obtaining query data
	for f in os.listdir(query_folder):
		im = Image.open(query_folder+f,'r')
		query = np.array(list(im.getdata()))
		query_data.append(depixelize(query))



	""" Run prediction and measure time taken"""
	""" Begin """


	time_start = datetime.now()

	#Normalising both training data and query data
	normalised_training_data = np.apply_along_axis(normalise,1,training_data)
	normalised_query_data = np.apply_along_axis(normalise,1,query_data[0:datasize])

	print("Data normalised")

	#Using PCA to reduce the dimensionality of the data
	pca = PCA.loadData(normalised_training_data)
	reduced_training_data = np.apply_along_axis(pca.reduce, 1, normalised_training_data)
	reduced_query_data = np.apply_along_axis(pca.reduce, 1, normalised_query_data)

	print("Dimensionality reduced")

	#Classifying using nearest neighbor
	kNN = kNearestNeighbor();
	kNN.train(reduced_training_data, training_labels)
	result =  kNN.predict(reduced_query_data, num_class) 

	print("Finished")

	""" Finish  """
	time_finished = datetime.now()
	duration = time_finished - time_start
	print("Time = "+ str(duration))

	#Write to file
	with open('query_output/output.csv', 'w') as csvfile:
	    writer = csv.writer(csvfile, delimiter=' ',
	                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
	    writer.writerow(result)
