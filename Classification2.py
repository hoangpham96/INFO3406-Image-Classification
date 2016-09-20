from Definitions import *

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
    normalised_training_data = np.apply_along_axis(normalise,1,training_data)
    normalised_test_data = np.apply_along_axis(normalise,1,test_data[0:datasize])

    print("Data normalised")

    #Using PCA to reduce the dimensionality of the data
    pca = PCA.loadData(normalised_training_data)
    reduced_training_data = np.apply_along_axis(pca.reduce, 1, normalised_training_data)
    reduced_test_data = np.apply_along_axis(pca.reduce, 1, normalised_test_data)

    print("Dimensionality reduced")

    #Classifying using nearest neighbor
    kNN = kNearestNeighbor();
    kNN.train(reduced_training_data,training_fine_label)
    result = kNN.predict(reduced_test_data, num_class)

    print("Fine lable assigned")

    #Classifying using nearest neighbor
    kNN2 = kNearestNeighbor();
    kNN2.train(reduced_training_data,training_coarse_label)
    result2 =  kNN2.predict(reduced_test_data, num_superclass)

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
