from Definitions import *
from Transformations import depixelize
import os
from PIL import Image

if __name__ == "__main__":
    #Define file paths
    meta_file = 'data2/meta'
    training_file = 'data2/train'
    query_folder = 'INFO3406_assignment1_query/'

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
    kNN.train(reduced_training_data,training_fine_label)
    result = kNN.predict(reduced_query_data, num_class)

    print("Fine lable assigned")

    #Classifying using nearest neighbor
    kNN2 = kNearestNeighbor();
    kNN2.train(reduced_training_data,training_coarse_label)
    result2 =  kNN2.predict(reduced_query_data, num_superclass)

    print("Coarse lable assigned")


    """ Finish  """
    time_finished = datetime.now()
    duration = time_finished - time_start
    print("Time = "+ str(duration))


    #Write to file
    with open('query2_output/output.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(result)
        writer.writerow(result2)
