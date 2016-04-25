from hearing_assistive import hear_assist_tools as ha

################################################################################################

# class Dataset:
	# represents a dataset of sound signals and its coresponding targets





################################################################################################

def load_data(filename):
	wave = ha.read_wave(filename)
	wave.normalize()
	return len(wave.ys)

def train(training_set=None, target_set=None):
	return