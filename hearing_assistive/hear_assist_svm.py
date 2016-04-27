import numpy as np
from sklearn import svm
from sklearn import linear_model

from hearing_assistive import hear_assist_tools as ha
from hearing_assistive import noise

################################################################################################
#
# GLOBAL FUNCTIONS
#
################################################################################################

def extract_feature(wave, seg_length=0.04):
	# extract the mfcc features of the given wave per segment
	#
	# wave: Wave object
	# seg_length: length of each segment in seconds
	#
	# return: matrix containing number of segments by coefficients (entire set will represent one feature in SVM)

	seg_samples = int(seg_length * wave.framerate) # number of samples per seg_length

	spectrogram = wave.make_spectrogram(seg_length=seg_samples)

	feature = spectrogram.mfcc()   # use default parameters, only include coefficients 2-14 (13 coefficients)

	return feature

def generate_feature_set(wave, target, setsize=100):
	# generate the training set (dataset and targetset) for an SVM, given a particular Wave object
	#
	# wave: Wave object
	# target: assigned target label
	# setsize: the number of samples to create (the size of the resulting training set)
	#
	# return: (dataset, targetset) dataset contains the mfcc features of each sample, targetset contains the target labels
	dataset = []
	targetset = []

	for i in range(setsize):
		# append clean element first
		if i == 0:
			feature = extract_feature(wave)		# extract mfccs from wave

			dataset.append(feature)
			targetset.append(target)

		# append elements with added noise
		else:
			new_wave = noise.apply_filter(wave) 	# apply a random noise filter to wave
			# feature = extract_feature(new_wave)		# extract mfccs from new_wave
			feature = extract_feature(wave)

			dataset.append(feature)
			targetset.append(target)

	# convert these sets into numpy arrays
	dataset = np.asarray(dataset)
	targetset = np.asarray(targetset)

	# return a Dataset object 
	return dataset, targetset

####################################################################################################################
#
# DATASET CLASS
#
####################################################################################################################

# class Dataset:
	# represents a dataset of sound signals and its coresponding targets