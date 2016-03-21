import numpy as np 
import matplotlib.pyplot as pyplot

import scipy
import scipy.stats
import scipy.fftpack
import struct
import subprocess

from fractions import gcd
from wave import open as open_wave

import warnings
try:
    from IPython.display import Audio
except:
    warnings.warn("Can't import Audio from IPython.display; Wave.make_audio() will not work.")

import array
import copy
import math

#################################################################################################


def read_wave(filename='sound.wav'):
    # Reads a wave file
    # filename: string
    # return: Wave

    fp = open_wave(filename, 'r')

    nchannels = fp.getnchannels()	# number of audio channels (1 for mono, 2 for stereo)
    nframes = fp.getnframes()		# number of audio frames
    sampwidth = fp.getsampwidth()	# sample width in bytes
    framerate = fp.getframerate()	# sampling frequency
    
    z_str = fp.readframes(nframes)	# reads and returns at most nframes of audio as a string of bytes
    
    fp.close()

    dtype_map = {1:np.int8, 2:np.int16, 3:'special', 4:np.int32}
    if sampwidth not in dtype_map:
        raise ValueError('sampwidth %d unknown' % sampwidth)
    
    if sampwidth == 3:
        xs = np.fromstring(z_str, dtype=np.int8).astype(np.int32)
        ys = (xs[2::3] * 256 + xs[1::3]) * 256 + xs[0::3]
    else:
        ys = np.fromstring(z_str, dtype=dtype_map[sampwidth])

    # if it's in stereo, just pull out the first channel
    if nchannels == 2:
        ys = ys[::2]

    #ts = np.arange(len(ys)) / framerate
    wave = Wave(ys, framerate=framerate)
    wave.normalize()

    return wave

def find_index(x, xs):
	# find the index corresponding to a given value in an array

	n = len(xs)
	start = xs[0]
	end = xs[-1]
	i = round((n-1) * (x - start) / (end - start))

	return int(i)

###########################################################################################


class Wave:
	# represents a discrete-time waveform

	def __init__(self, ys, ts=None, framerate=None):
		# initializes the wave
		# 
		# ys: wave array
		# ts: array of times
		# framerate: samples per seconds

		self.ys = np.asanyarray(ys)
		self.framerate = framerate if framerate is not None else 11025

		if ts is None:
			self.ts = np.arange(len(ys)) / float(self.framerate)
		else:
			self.ts = np.asanyarray(ts)

   	def copy(self):
        # makes a copy.
		# return: new Wave
		
		return copy.deepcopy(self)
	
	def __len__(self):
		return len(self.ys)

	@property
	def start(self):
	    return self.ts[0]

	@property
	def end(self):
	    return self.ts[-1]

	@property
	def duration(self):
		# duration (property)
		#
		# return: float duration in seconds

	    return len(self.ys) / float(self.framerate)

	#####################################################################

	def find_index(self, t):
		# find the index corresponding to a given time
	   	
	    n = len(self)
	    start = self.start
	    end = self.end
	    i = round((n-1) * (t - start) / (end - start))

	    return int(i)

	def segment(self, start=None, duration=None):
		# extracts a segment
		# 
		# start: float start time in seconds
		# duration: float duration in seconds
		#
		# return: Wave
	    
	    if start is None:
	        start = self.ts[0]
	        i = 0
	    else:
	        i = self.find_index(start)

		j = None if duration is None else self.find_index(start + duration)

	    return self.slice(i, j)

	def slice(self, i, j):
		# makes a slice from a Wave
		#
		# i: first slice index
		# j: seconds slice index
	    
	    ys = self.ys[i:j].copy()
	    ts = self.ts[i:j].copy()
	    return Wave(ys, ts, self.framerate)

	def make_spectrum(self, full=False):
		# computes the spectrum using fast fourier transform (FFT)
		#
		# return: Spectrum

		n = len(self.ys)
		d = 1 / float(self.framerate)

		if full:
			hs = np.fft.fft(self.ys)
			fs = np.fft.fftfreq(n, d)
		else:
			hs = np.fft.rfft(self.ys)
			fs = np.fft.rfftfreq(n, d)

		return Spectrum(hs, fs, self.framerate, full)

	####################################################################

	def normalize(self, amp=1.0):
		# normalize the signal to the given amplitude
		#
		# amp: float amplitude
	    
	    self.ys = normalize(self.ys, amp=amp)

	def apodize(self, denom=20, duration=0.1):
		# tapers the amplitude at the beginning and end of the signal
		#
		# tapers either the given duration of time or the given fraction of the total duration, whichever is less
		#
		# denom: float fraction of the segment to taper
		# duration: float duration of the taper in seconds

		self.ys = apodize(self.ys, self.framerate, denom, duration)

	def stretch(self, factor):
		# modifies the speed of the wave by a factor

		self.framerate = self.framerate * factor
		self.ts = np.arange(len(self.ys)) / self.framerate

	############################################################################

	def plot(self):
		# plots the wave

		time = np.linspace(self.start, self.duration, len(self.ys))	# get seconds

		#pyplot.plot(self.ys)
		pyplot.plot(time, self.ys, color='#5F9EA0')
		pyplot.show()

	def make_audio(self):
		# makes an IPython Audio object

		audio = Audio(data=self.ys.real, rate=self.framerate)
		return audio

############################################################################


class _SpectrumParent:
	# contains code common to Spectrum

	def __init__(self, hs, fs, framerate, full=False):
		# initializes the Spectrum
		#
		# hs: array of amplitudes (real of complex)
		# fs: array of frequencies
		# framerate: frames per second
		# full: boolean to indicate full or real FFT

		self.hs = np.asanyarray(hs)
		self.fs = np.asanyarray(fs)
		self.framerate = framerate
		self.full = full

	@property
	def max_freq(self):
		# return the Nyquist frequency for this spectrum

		return self.framerate / 2

	@property
	def amps(self):
		# return a sequence of amplitudes (read-only property)

		return np.absolute(self.hs)

	def render_full(self, high=None):
		# extracts amps and fs from a full spectrum
		#
		# high: cutoff frequency
		#
		# return: fs, amps

		hs = np.fft.fftshift(self.hs)
		amps = np.abs(hs)
		fs = np.fft.fftshift(self.fs)
		i = 0 if high is None else find_index(-high, fs)
		j = None if high is None else find_index(high, fs) + 1

		return fs[i:j], amps[i:j]

	def plot(self, high=None):
		# plots amplitude vs frequency
		#
		# if full spectrum, ignore low and high
		#
		# high: frequency to cut off at
	
		if self.full:
			fs, amps = self.render_full(high)
			pyplot.plot(fs, amps, color='#5F9EA0')
			pyplot.show()
		else:
			i = None if high is None else find_index(high, self.fs)
			pyplot.plot(self.fs[:i], self.amps[:i], color='#5F9EA0')
			pyplot.show()

####################################################################################


class Spectrum(_SpectrumParent):
	# represents the spectrum of a signal

	def __len__(self):
		# length of the spectrum

		return len(self.hs)

	def low_pass(self, cutoff, factor=0):
		# attenuate frequencies above the cutoff
		#
		# cutoff: frequency in Hz
		# factor: multiplied to magnitude

		self.hs[abs(self.fs) > cutoff] *= factor

	def high_pass(self, cutoff, factor=0):
		# attenuate frequencies below the cutoff
		#
		# cutoff: frequency in Hz
		# factor: multiplied to magnitude

		self.hs[abs(self.fs) < cutoff] *= factor

	def band_stop(self, low_cutoff, high_cutoff, factor=0):
		# attenuate frequencies between the cutoffs
		#
		# low_cutoff: frequency in Hz
		# high_cutoff: frequency in Hz
		# factor: multiplied to magnitude

		fs = abs(self.fs)
		indices = (low_cutoff < fs) & f(fs < high_cutoff)
		self.hs[indices] *= factor

	################################################################################

	def make_wave(self):
		# transforms to the time domain
		#
		# return: Wave

		if self.full:
			ys = np.fft.ifft(self.hs)
		else:
			ys = np.fft.irfft(self.hs)

		return Wave(ys, framerate=self.framerate)

#################################################################################################


def normalize(ys, amp=1.0):
	# normalize a wave array so the maximum amplitude is +amp or -amp
	#
	# ys: wave array
	# amp: max amplitude (pos or neg) in result
	#
	# return: wave array
    
    high = abs(max(ys))
    low = abs(min(ys))

    return amp * ys / max(high, low)

def apodize(ys, framerate, denom=20, duration=0.1):
	# tapers the amplitude at the beginning and end of the signal.
	#
	# tapers either the given duration of time or the given fraction of the total duration, whichever is less
	#
	# ys: wave array
	# framerate: int frames per second
	# denom: float fraction of the segment to taper
	# duration: float duration of the taper in seconds
	#
	# return: wave array

    # a fixed fraction of the segment
    n = len(ys)
    k1 = n // denom

    # a fixed duration of time
    k2 = int(duration * framerate)

    k = min(k1, k2)

    w1 = np.linspace(0, 1, k)
    w2 = np.ones(n - 2*k)
    w3 = np.linspace(1, 0, k)

    window = np.concatenate((w1, w2, w3))
    return ys * window