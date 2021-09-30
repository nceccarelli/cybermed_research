# Respiratory Sound Detection Scripts

## Scripts

### keras_resnet_crackling.py
This script runs the resnet model over the data for crackling detection. 
The data_folder variable is expected to be the path to the pre-processed data.

### keras_resnet_wheezing.py
Same as above, but for wheezing.

### pre_process_keras.py
This script runs the preprocessing to separate the samples into training and testing files, generate spectrograms, and create the ground truth CSV files.
The input_data variable should be a path to a folder that contains the audio and text annotation files.
The output_data variable shoudl be a path to a folder that is where you would like the spectrograms and csv files to be located. 
**NOTE:** within this folder, there needs to be folders named "training" and "testing".

### pre_process_keras_LOO.py
The same as the above, but with Leave-One-Out cross validation.

### pre_process_keras_butterworth.py
The same as the above, but without Leave-One-Out cross validation, and with a Butterworth filter.

### run_augmentation.py
This script runs the data augmentation described in the paper.
It is a wrapper for `spec_augment_tensorflow.py`.
There are several parameters here, all can be seen within the code in the argument parser logic at the top of the script. 
Of note, the `--raw-path` argument is the data folder containing the pre-processed data.
The default values for the arguments within the argument parser logic are the parameters I used. 

### spec_augment_tensorflow.py
You do not need to use this script. `run_augmentation.py` does is a wrapper for this script.

## Evaluation Files
These files were graphic depictions of the data we were working with. 
`original_PDFs/` contains the graphs and pictures included in the Overleaf document.
`sample_spectrograms/` contains some canonical spectrograms that vary in appearance; they were included in the paper as well.
`evaluation_charts.xlsx` is an Excel document with all of the data I collected manually in my experimenation. 
It is a little messy, so let me know if you have questions.