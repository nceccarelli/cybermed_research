# File processing
import os
import fnmatch
import csv

# Math
import numpy as np

# Plotting
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Audio processing
import librosa
import librosa.display
from pydub import AudioSegment

# Image processing
from PIL import Image

# For set_size()
from matplotlib.image import imread
from tempfile import NamedTemporaryFile

# GitHub repo from Google
# import spec_augment_tensorflow


# Directory storing input audio files
input_dir = "../database/audio_and_txt_files/"

# Directory where you want the spectrograms stored
output_dir = "../database/pre_processed_data_keras/"

max_time = 3.5
min_time = 0.5
keep_time = 2.0
half_keep_time = keep_time/2.0
freq_min = 100
freq_max = 2000

num_pixels = 80388
skipped_count = 0

sample_count = 0 # keeps track of current sample
placement_flag = "training" # keeps track of training vs testing data

training_crackling_truth = [] # store for training data, later stored to CSV
training_wheezing_truth = []
testing_crackling_truth = [] #  store for testing data, later stored to CSV
testing_wheezing_truth = []

patient = 1 # keeps track of current patient

def get_size(fig, dpi=100):
    with NamedTemporaryFile(suffix='.png') as f:
        fig.savefig(f.name, bbox_inches='tight', dpi=dpi)
        height, width, _channels = imread(f.name).shape
        return width / dpi, height / dpi

def set_size(fig, size, dpi=100, eps=1e-2, give_up=2, min_size_px=10):
    target_width, target_height = size
    set_width, set_height = target_width, target_height # reasonable starting point
    deltas = [] # how far we have
    while True:
        fig.set_size_inches([set_width, set_height])
        actual_width, actual_height = get_size(fig, dpi=dpi)
        set_width *= target_width / actual_width
        set_height *= target_height / actual_height
        deltas.append(abs(actual_width - target_width) + abs(actual_height - target_height))
        if deltas[-1] < eps:
            return True
        if len(deltas) > give_up and sorted(deltas[-give_up:]) == deltas[-give_up:]:
            return False
        if set_width * dpi < min_size_px or set_height * dpi < min_size_px:
            return False

for file in os.listdir(input_dir): #loop through all files in input directory
    if fnmatch.fnmatch(file, "*.wav"): #if its a WAV
        root_file = file[0:len(file)-3] #get file name minus "WAV"
        txt_file = file[0:len(file)-3] + "txt" # add "wav" for audio, "txt" for info
        print(txt_file)
        with open(input_dir + txt_file) as tsv: # open associated TXT file

            for line in csv.reader(tsv, delimiter="\t"): #iterate through each line of the text file

                # read start and end time from file
                recorded_start_time = float(line[0])
                recorded_end_time = float(line[1])

                # Do not process respirations longer than 4 seconds or less than 1 second in duration
                if (recorded_end_time - recorded_start_time > max_time or recorded_end_time - recorded_start_time < min_time): continue

                # Mid-point of recording
                middle_start_time = recorded_start_time + (recorded_end_time - recorded_start_time)
                
                # Standardize data to 2.8 seconds in length as best as possible
                start_time = (middle_start_time - keep_time/2) * 1000
                end_time = (middle_start_time + keep_time/2) * 1000

                # make sure start time is not negative
                if start_time < 0: 
                    start_time = 0
                    if (recorded_end_time >= keep_time):
                        end_time = keep_time*1000

                # Make sure end time is not over the length of the respiration
                elif end_time / 1000 > recorded_end_time: 
                    start_time = (recorded_end_time - keep_time) * 1000
                    end_time = recorded_end_time * 1000
                    if start_time < 0: start_time = 0
    
                # Skip respiration if regularization of data fails
                if (end_time - start_time != 2000): continue

                # Record next sample
                sample_count += 1

                # figure out if it's testing or training
                if (not sample_count % 5): placement_flag = "testing" # assign every fifth to testing
                else: placement_flag = "training" # assign four of five to training

                # Update status to user
                print("Processing sample", str(sample_count), "| From patient", patient)

                # read if the respiration crackles or wheezes
                crackles = int(line[2])
                wheezes = int(line[3])

                # # get WAV and segment it
                sep_audio = AudioSegment.from_wav(input_dir + file) #get audio file
                sep_audio = sep_audio[start_time:end_time] #separate time
                sep_audio.export('intermediate1.wav', format='wav')

                # Load WAV and create spectrogram
                y,sr = librosa.load("intermediate1.wav")
                S = librosa.feature.melspectrogram(y=y, sr=sr,fmin=freq_min,fmax=freq_max)
                S_dB = librosa.power_to_db(S, ref=np.max)
                librosa.display.specshow(S_dB, x_axis='time',
                                        y_axis='mel', sr=sr,
                                        fmin=freq_min,fmax=freq_max)
                
                # Save spectrogram
                plt.axis("off")
                fig = plt.gcf()
                fig.set_size_inches(8, 8)
                sample_save_dir = output_dir + placement_flag + '/sample_' + str(sample_count) + '.png'
                set_size(fig, (2.44,2.44))
                plt.savefig(sample_save_dir, bbox_inches="tight", pad_inches = 0)

                # Open image for quality check
                # image = Image.open(sample_save_dir)
                # im_arr = np.asarray(image)

                # run quality check
                # peak_pix = 0
                # for column in im_arr:
                #     for (r,g,b,_) in column:
                #         r,g,b = int(r),int(g),int(b)
                #         if b*2-40 < (r+g): peak_pix += 1
                        
                # If there is too much yellow, skip. Quality check declined
                # if (peak_pix/num_pixels) > 0.4: 
                # # Quality check failed
                #     os.remove(sample_save_dir)
                #     print("Sample", str(sample_count), "skipped.")
                #     sample_count -= 1
                #     skipped_count += 1
                # else:
                # Quality check passed
                    # Run data augmentation

                    # # Create spectrogram
                    # mel_spectrogram = librosa.feature.melspectrogram(y=y,
                    #                                     sr=sr,
                    #                                     n_mels=256,
                    #                                     hop_length=128,
                    #                                     fmax=freq_max)
                    # S_dB = librosa.power_to_db(mel_spectrogram, ref=np.max)
                    # librosa.display.specshow(S_dB, x_axis='time',
                    #                         y_axis='mel', sr=sr,
                    #                         fmin=freq_min,fmax=freq_max)
                    
                    # # Save augmented spectrogram
                    # plt.axis("off")
                    # fig = plt.gcf()
                    # fig.set_size_inches(4.5, 3)
                    # augmented_sample_save_dir = output_dir + placement_flag + '/sample_' + str(sample_count) + '-1.png'
                    # plt.savefig(augmented_sample_save_dir, bbox_inches="tight", pad_inches = 0)
                    
                    # append the ground truth to corresponding list
                if (placement_flag == 'testing'):
                    testing_crackling_truth.append(crackles)
                    testing_wheezing_truth.append(wheezes)
                else: 
                    training_crackling_truth.append(crackles)
                    training_wheezing_truth.append(wheezes)
            
                # Remove intermediate file
                os.remove("intermediate1.wav")

                print(len(training_crackling_truth)+len(testing_crackling_truth))

        #increment patient 
       
        patient += 1

# write truth training data to CSV
with open(output_dir + 'training_crackling_truth.csv', 'w') as training_file:
    wr = csv.writer(training_file)
    wr.writerow(training_crackling_truth)

with open(output_dir + 'training_wheezing_truth.csv', 'w') as training_file:
    wr = csv.writer(training_file)
    wr.writerow(training_wheezing_truth)

# write truth testing data to CSV
with open(output_dir + 'testing_crackling_truth.csv', 'w') as testing_file:
    wr = csv.writer(testing_file)
    wr.writerow(testing_crackling_truth)

# write truth testing data to CSV
with open(output_dir + 'testing_wheezing_truth.csv', 'w') as testing_file:
    wr = csv.writer(testing_file)
    wr.writerow(testing_wheezing_truth)


print("Skipped count: ", str(skipped_count))