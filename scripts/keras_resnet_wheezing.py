import tensorflow as tf
import csv
import numpy as np
import os
import imageio
from keras.utils import to_categorical
from keras import backend as K

# Folder to read in the data from
    # Folder should contain:
        # Training and testing folders with spectrograms
        # 4 CSVs with training and testing data for both crackling and wheezing
data_folder = "../database/pre_processed_data_keras_butterworth/"
lung_problem_detection = "wheezing"

# Get the number of samples
def get_spectro_counts():
    # Get number of training spectrograms
    _, _, files = next(os.walk(data_folder + "training"))
    training_png_count = 0
    for file in files:
        if "aug" not in file:
            training_png_count += 1

    # Get number of testing spectrograms
    _, _, files = next(os.walk(data_folder + "testing"))
    testing_png_count = 0
    for file in files:
        if "aug" not in file:
            testing_png_count += 1

    return training_png_count + testing_png_count

# Read in the data
def read_data():
    x_train = []
    x_test = []
    y_train = []
    y_test = []

    repeats = []

    # Get the total number of files
    tot_files = get_spectro_counts()
    print("Loading data. Files to process: ", str(tot_files))

    # Get x_test
    i = 5
    while i <= tot_files:
        im = imageio.imread(data_folder + "testing/sample_" + str(i) + ".png")
        x_test.append(im[:,:,0:3])

        # Check for augmented image
        if (os.path.exists(data_folder + "testing/sample_" + str(i) + "_aug.png")):
            # Augmented image exists
            repeats.append(i)

            # Load and saveimage
            im = imageio.imread(data_folder + "testing/sample_" + str(i) + "_aug.png")
            x_test.append(im[:224,:224,0:3])

        i += 5
    x_test = np.array(x_test)

    # Get x_train
    i = 1
    while i <= tot_files:
        im = imageio.imread(data_folder + "training/sample_" + str(i) + ".png")
        x_train.append(im[:,:,0:3])

        # Check for augmented image
        if (os.path.exists(data_folder + "training/sample_" + str(i) + "_aug.png")):
            # Augmented image exists
            repeats.append(i)

            # Load and saveimage
            im = imageio.imread(data_folder + "training/sample_" + str(i) + "_aug.png")
            
            # TODO: This is a stopgap, must rescale, not resize
            x_train.append(im[:224,:224,0:3])

        if ((i+1)%5==0): i += 2
        else: i += 1
    x_train = np.array(x_train)

    # Get y_train
    with open(data_folder + "training_" + lung_problem_detection + "_truth.csv", 'r') as tcsv:
        for line in csv.reader(tcsv):
            i = 1
            for elem in line:
                y_train.append(int(elem))
                if (i in repeats): y_train.append(int(elem))

                if ((i+1)%5==0): i += 2
                else: i += 1
    y_train = np.asarray(y_train)
    
    # Get y_test
    with open(data_folder + "testing_" + lung_problem_detection + "_truth.csv", 'r') as tcsv:
        for line in csv.reader(tcsv):
            i = 5
            for elem in line:
                y_test.append(int(elem))

                if (i in repeats): y_test.append(int(elem))
                i += 5
                
    y_test = np.asarray(y_test)

    print("Fininshed loading data")

    print("X train", x_train.shape)
    return (x_train, y_train), (x_test, y_test)

# Calculate recall
def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

# Calculate precision
def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

# Calculate F-1 score
def f1(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2*((prec*rec)/(prec+rec+K.epsilon()))

# Create ResNet pretrained on ImageNet
model = tf.keras.applications.ResNet50(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000
)

# I may be able to add a keras LSTM layer here

# Configure learning rate for optimizer
# optimizer = tf.keras.optimizers.Adam(0.001)
opt = tf.keras.optimizers.Adam(0.001)
# optimizer.learning_rate.assign(0.01)

# Compile the model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy', f1, precision, recall]
)

# Read in the data
(x_train, y_train), (x_test, y_test) = read_data()

# Run model
model.fit(
    x_train, y_train, 
    batch_size = 32, 
    epochs = 10, 
    verbose = 1 
)

# Evaluate model
results = model.evaluate(
    x_test, y_test, 
    batch_size=128
)