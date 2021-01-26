import sys
import keras
import numpy as np
from keras.models import Model
from tensorflow.keras.models import load_model
from classification_functions import MnistDataloaderForImages, MnistDataSaverForImages


if __name__ == "__main__":
    if (len(sys.argv) == 11):
        i = 0
        for var in sys.argv:  # get values from command line
            if (var == "-d"):
                training_images_filepath = sys.argv[i + 1]
            if (var == "-od"):
                training_images_filepath_output = sys.argv[i + 1]
            if (var == "-q"):
                test_images_filepath = sys.argv[i + 1]
            if (var == "-oq"):
                test_images_filepath_output = sys.argv[i + 1]
            i = i + 1
    else:
        print("Wrong input. Using default values.")
        training_images_filepath = 'train-images-idx3-ubyte'  # default values if not given by user
        training_images_filepath_output = 'train-images-idx1-ushort'
        test_images_filepath = 't10k-images-idx3-ubyte'
        test_images_filepath_output = 't10k-images-idx1-ushort'
        model = 'reduce_model.h5'

    x_train = MnistDataloaderForImages(training_images_filepath)  # read datasets
    x_test = MnistDataloaderForImages(test_images_filepath)  # read datasets

    x_train = np.array(x_train)
    x_test = np.array(x_test)

    x_train = x_train.astype('float32') / 255.  # values 0/1
    x_test = x_test.astype('float32') / 255.

    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

    autoencoder_model = load_model(model)  # load autoencoder model
    l = (len(autoencoder_model.layers) / 2) - 1

    output = autoencoder_model.layers[int(l)].output

    reducer_model = Model(inputs=autoencoder_model.layers[0].output, outputs=output, name='REDUCER')

    reducer_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    reducer_model.summary()

    # training file
    print("converting ... " + training_images_filepath + " to " + training_images_filepath_output)
    images_10_train = reducer_model.predict(x_train)

    # write to disk ....
    maxvalue = np.amax(images_10_train)
    minvalue = np.amin(images_10_train)

    # y = (x+min)/(max+abs(min))*25500
    scale = 25500/(maxvalue - minvalue)
    images_10_train = (images_10_train - minvalue)*scale

    MnistDataSaverForImages(images_10_train, training_images_filepath_output)

    # test file
    print("converting ... " + test_images_filepath + " to " + test_images_filepath_output)
    images_10_test = reducer_model.predict(x_test)

    # write to disk ....
    maxvalue = np.amax(images_10_test)
    minvalue = np.amin(images_10_test)

    # y = (x+min)/(max+abs(min))*25500
    scale = 25500/(maxvalue - minvalue)
    images_10_test = (images_10_test - minvalue)*scale

    MnistDataSaverForImages(images_10_test, test_images_filepath_output)


