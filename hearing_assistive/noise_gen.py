from hearing_assistive import hear_assist_tools as ha

import numpy as np

####################################################################################

def add_wave(wave1, wave2):
	# add two waves
	#
	# return: new Wave

	assert wave1.framerate == wave2.framerate

	# make an array of times that covers both waves
	start = min(wave1.start, wave2.start)
	end = max(wave1.end, wave2.end)
	n = int(round((end - start) * wave1.framerate)) + 1
	ys = np.zeros(n)
	ts = start + np.arange(n) / wave1.framerate

	def add_ys(wave):
		i = ha.find_index(wave.start, ts)
		diff = ts[i] - wave.start
		dt = 1 / wave.framerate
		if (diff / dt) > 0.1:
			warnings.warn("Cannot add these waveforms, their time arrays do not line up.")
		j = i + len(wave)
		ys[i:j] += wave.ys

	add_ys(wave1)
	add_ys(wave2)

	return ha.Wave(ys, ts, wave1.framerate)

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

	def plot(self, framerate=44100):
		# plots the signal
		#
		# the default behavior is to plot three periods
		# 
		# framerate: samples per second

		duration = self.period * 3
		wave = self.make_wave(duration, start=0, framerate=framerate)
		wave.plot()

	def make_wave(self, duration=1, start=0, framerate=44100):
		# makes a Wave object
		#
		# duration: float seconds
		# start: float seconds
		# framerate: int frames per second
		#
		# return: Wave

		n = round(duration * framerate)
		ts = start + np.arange(n) / framerate
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

class PinkNoise(_Noise):
	# represents Pink noise

	def __init__(self, amp=1.0, beta=1.0):
		# initializes a pink noise signal
		#
		# amp: float amplitude, 1.0 is nominal max

		self.amp = amp
		self.beta = beta

	def make_wave(self, duration=1, start=0, framerate=44100):
		# make a Wave object
		#
		# duration: float seconds
		# start: float seconds
		# framerate: int frames per second
		#
		# return: Wave

		signal = UncorrelatedUniformNoise()
		wave = signal.make_wave(duration, start, framerate)
		spectrum = wave.make_spectrum()

		denom = spectrum.fs ** (self.beta / 2.0)
		denom[0] = 1
		spectrum.hs /= denom

		wave2 = spectrum.make_wave()
		wave2.ys = ha.unbias(wave2.ys)
		wave2.ys = ha.normalize(wave2.ys, self.amp)

		return wave2