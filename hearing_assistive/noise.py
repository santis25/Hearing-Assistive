from hearing_assistive import hear_assist_tools as ha

import numpy as np
import random

############################################################################################################################
#
# GLOBAL FUNCTIONS
#
#############################################################################################################################

def generate_intensity():
	# generate a random intensity level for a noise filter
	#
	# return: float

	noise_intensity = random.uniform(0.01, 0.02)  	# generate a random float number between 0.1 and 0.5
	return noise_intensity

def apply_filter(wave, factor=1):
	# randomly select a noise filter and apply
	#
	#
	# KEY | NOISE FILTER
	# ---------------------
	# 0   | Uncorrelated Uniform Noise
	# 1   | Uncorrelated Gaussian Noise
	# 2   | Brownian Noise (Red Noise)
	#
	# wave: Wave object
	#
	# return: new Wave

	noise_filter = random.randint(0,2)  # generate a random integer between 0 and 2, inclusive (including endpoints)
	noise_intensity = factor * generate_intensity()	# generate noise intensity

	# obtain corresponding noise signal
	if noise_filter == 0:
		signal = UncorrelatedUniformNoise(amp=noise_intensity)
	elif noise_filter == 1:
		signal = UncorrelatedGaussianNoise(amp=noise_intensity)
	elif noise_filter == 2:
		signal = BrownianNoise(amp=noise_intensity)

	# make noise wave
	noise = signal.make_wave(duration=wave.duration, start=wave.start)

	# apply the noise filter to the original wave (add the two waves together)
	new_wave = add_wave(wave, noise)
	new_wave.normalize()

	return new_wave


def add_wave(wave1, wave2):
	# add two waves
	#
	# NOTE: this operation ignores the timestamps, the result has the timestamps of wave1
	#
	# return: new Wave

	assert wave1.framerate == wave2.framerate
	assert len(wave1) == len(wave2)

	ys = wave1.ys + wave2.ys

	return ha.Wave(ys, wave1.ts, wave1.framerate)

##############################################################################################
#
# SIGNAL CLASS
#
##############################################################################################

class Signal:
	# represents a time-varying signal

	@property
	def period(self):
		# period of the signal in seconds (property)
		#
		# return: float seconds

	    return 0.1

	def plot(self, title=None, framerate=44100):
		# plots the signal
		#
		# the default behavior is to plot three periods
		# 
		# framerate: samples per second

		duration = self.period * 3
		wave = self.make_wave(duration, start=0.0, framerate=framerate)
		wave.plot(title)

	def make_wave(self, duration=1.0, start=0.0, framerate=44100):
		# makes a Wave object
		#
		# duration: float seconds
		# start: float seconds
		# framerate: int frames per second
		#
		# return: Wave

		n = round(duration * framerate)
		ts = start + np.arange(n) / float(framerate)
		ys = self.evaluate(ts)

		return ha.Wave(ys, ts, framerate=framerate)

##############################################################################################
#
# _NOISE CLASS
#
##############################################################################################

class _Noise(Signal):
	# represents a noise signal

	def __init__(self, amp=1.0):
		# initializes a white noise signal
		#
		# amp: float amplitude, 1.0 is nominal max

		self.amp = amp

##############################################################################################
#
# NOISE CLASSES
#
##############################################################################################

class UncorrelatedUniformNoise(_Noise):
	# represents uncorrelated uniform noise

	def evaluate(self, ts):
		# evaluates the signal at the given times
		#
		# ts: float array of times
		#
		# return: float wave array

		ys = np.random.uniform(-self.amp, self.amp, len(ts))
		return ys

class UncorrelatedGaussianNoise(_Noise):
	# represents uncorrelated gaussian noise

	def evaluate(self, ts):
		# evaluates the signal at the given times
		#
		# ts: float array of times
		#
		# return: float wave array

		ys = np.random.normal(0, self.amp, len(ts))
		return ys

class BrownianNoise(_Noise):
	# represents Brownian noise, also called red noise

	def evaluate(self, ts):
		# evaluates the signal at the given times
		#
		# computes Brownian noise by taking the cumaltive sum of a uniform random series
		#
		# ts: float array of times
		#
		# return: float wave array

		dys = np.random.uniform(-1, 1, len(ts))
		ys = np.cumsum(dys)
		ys = ha.normalize(ha.unbias(ys), self.amp)

		return ys