from Classification import *

if __name__ == "__main__":
    #Define file paths
    meta_file = 'data2/meta'
    training_file = 'data2/train'
    test_file = 'data2/test'

    #Obtaining lable names
    meta = unpickle(meta_file)
    class_names = meta['fine_label_names']
    superclass_names = meta['coarse_label_names']
    num_class = len(class_names)
    num_superclass = len(superclass_names)


    # #Obtaining training data and lables
    train = unpickle(training_file)
    training_data = train['data']
    training_fine_label = train['fine_labels']
    training_coarse_label = train['coarse_labels']

    # #Obtaining test data
    test = unpickle(test_file)
    test_data = test['data']



    """ Run prediction and measure time taken"""
    """ Begin """


    time_start = datetime.now()

    #Normalising both training data and test data
    normalised_training_data = []
    normalised_test_data = []
    for i in range(datasize*5):
	   normalised_training_data.append( normalise(training_data[i]) )

    for i in range(datasize):
    	normalised_test_data.append( normalise(test_data[i]) )
    normalised_training_data = np.array(normalised_training_data)
    normalised_test_data = np.array( normalised_test_data )

    print("Data normalised")

    #Classifying using nearest neighbor
    kNN = kNearestNeighbor();
    kNN.train(normalised_training_data,training_fine_label[0:datasize*5])
    result = kNN.predict(normalised_test_data, num_class)

    print("Fine lable assigned")

    #Classifying using nearest neighbor
    kNN2 = kNearestNeighbor();
    kNN2.train(normalised_training_data,training_coarse_label[0:datasize*5])
    result2 =  kNN2.predict(normalised_test_data, num_superclass)

    print("Coarse lable assigned")


    """ Finish  """
    time_finished = datetime.now()
    duration = time_finished - time_start
    print("Time = "+ str(duration))

    #Write to file
    with open('output2/output.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(result)
        writer.writerow(result2)
