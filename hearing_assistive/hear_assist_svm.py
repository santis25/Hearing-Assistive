import numpy as np
import sys
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

	feature = spectrogram.mfcc()   # use default parameters, *only include coefficients 2-14 (13 coefficients)
	
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

	assert wave.framerate == 44100

	print "Training New Dataset"

	for i in range(setsize):
		# append clean element first
		if i == 0:
			feature = extract_feature(wave)		# extract mfccs from wave

			dataset.append(feature)
			targetset.append(target)

		# append elements with added noise
		else:
			new_wave = noise.apply_filter(wave) 	# apply a random noise filter to wave
			feature = extract_feature(new_wave)		# extract mfccs from new_wave
			# feature = extract_feature(wave)

			dataset.append(feature)
			targetset.append(target)

		sys.stdout.write('#')    # show progress

	print '\n'

	# convert these sets into numpy arrays
	# dataset = np.asarray(dataset)
	# targetset = np.asarray(targetset)

	# return a Dataset object 
	return dataset, targetset

# def init_classifier():
# 	#

# 	clf = linear_model.SGDClassifier(n_iter=100, alpha=0.01)		# linear SVM classifier with Stochastic Gradient Descent training

# 	dataset = []
# 	targetset = []

# 	for i in range(100):
# 		# add dummy set of zeros to the dataset
# 		dummyset = np.zeros((119, 13))
# 		dataset.append(dummyset)
# 		targetset.append(0)
	    
# 	    # add dummy set of ones to the dataset
# 		dummyset = np.ones((119, 13)) 
# 		dataset.append(dummyset)
# 		targetset.append(1)

# 	    # add dummy set of negative ones to the dataset
# 		dummyset = np.negative(np.ones((119, 13)))
# 		dataset.append(dummyset)
# 		targetset.append(2)
	    
# 	# convert to numpy arrays
# 	dataset = np.asarray(dataset)
# 	targetset = np.asarray(targetset)

# 	# restructure dataset before being processed by svm
# 	dataset = dataset.reshape((dataset.shape[0], -1))

# 	# train dummyset
# 	clf.fit(dataset, targetset)

# 	return clf

def init_set(size):
	# initialize dataset and targetset 
	#
	# size: the number of datasets to generate per target
	#
	# return: dataset list, targetset list 

	dataset = []
	targetset = []

	for i in range(size):
		# add dummy set of zeros to the dataset
		dummyset = np.zeros((119, 26))
		dataset.append(dummyset)
		targetset.append(0)
	    
	    # add dummy set of ones to the dataset
		dummyset = np.ones((119, 26)) 
		dataset.append(dummyset)
		targetset.append(1)

	    # add dummy set of negative ones to the dataset
		dummyset = np.negative(np.ones((119, 26)))
		dataset.append(dummyset)
		targetset.append(2)

	return dataset, targetset

def append_set(basedata, basetarget, appenddata, appendtarget):
	# append two lists together
	#
	# return: new_dataset list, new_targeset list

	# initialize new dataset and targetset
	new_basedata = basedata
	new_basetarget = basetarget

	for i in range(len(appenddata)):
		new_basedata.append(appenddata[i])
		new_basetarget.append(appendtarget[i])

	return new_basedata, new_basetarget

def clf_accuracy(expected, actual):
	#

	assert len(expected) == len(actual)

	correct = 0
	for i in range(len(actual)):
		if expected[i] == actual[i]:
			correct += 1

	accuracy = (float(correct) / len(actual)) * 100

	return accuracy

def init_classifier(base=False, *waves):
	#
	
	clf = svm.SVC(kernel='rbf', C=100)		# initialize SVM classifier (kernel = radial basis function, C = penalty of error term)

	dataset = []
	targetset = []
	target = 0

	if base is True:
		dataset, targetset = init_set(100)		# initialize 3 basic datasets and targetsets (targets equal 0, 1, and 2)
		target = 3    							# set target to now be 3 for when continue to add other wave objects

	wavelist = list(waves)  				# place the wave objects into a list

	# train each wave object by generating 100 samples each with varying noise levels
	for wave in wavelist:
		new_dataset, new_targetset = generate_feature_set(wave, target)						# extract the mfcc features of wave
		dataset, targetset = append_set(dataset, targetset, new_dataset, new_targetset)	 	# append the new dataset and targetset
		target += 1

	# convert to numpy arrays
	dataset = np.asarray(dataset)
	targetset = np.asarray(targetset)

	print dataset.shape		# TEST

	# restructure dataset before being processed by svm
	dataset = dataset.reshape((dataset.shape[0], -1))

	# train dummyset
	clf.fit(dataset, targetset)

	return clf

####################################################################################################################
#
# DATASET CLASS
#
####################################################################################################################

# class Dataset:
	# represents a dataset of sound signals and its coresponding targets