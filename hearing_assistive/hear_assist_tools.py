import numpy as np 
import matplotlib
import matplotlib.pyplot as pyplot
import collections

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
		self.framerate = framerate if framerate is not None else 44100	# changed from 11025

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

		n = len(self.ys)					# number of samples
		d = 1 / float(self.framerate)		# inverse of the framerate, which is the time between samples

		# each value in hs corresponds to a frequency compenent -- its magnitude is proportional to the amplitude
		# of the corresponding component, its angle is the phase offset.
		if full:
			hs = np.fft.fft(self.ys)
			fs = np.fft.fftfreq(n, d)
		else:
			hs = np.fft.rfft(self.ys)
			fs = np.fft.rfftfreq(n, d)

		return Spectrum(hs, fs, self.framerate, full)

	def make_dct(self):
		# compute the discrte cosine transform of this wave
		#
		# return: Dct

		n = len(self.ys)
		hs = scipy.fftpack.dct(self.ys, type=2)
		fs = (0.5 + np.arange(n)) / 2

		return Dct(hs, fs, self.framerate)

	def make_spectrogram(self, seg_length, win_flag=True):
		# computes the spectrogram of the wave
		#
		# seg_length: number of samples in each segment
		# win_flag: boolean, whether to apply hamming window to each segment
		#
		# return: Spectrogram

		if win_flag:
			window = np.hamming(seg_length)		# sequence of multipliers that are the same length as the wave segment

		i = 0
		j = seg_length
		step = seg_length / 2

		# map from time to spectrum   
		spec_map = {}

		while j < len(self.ys):
			segment = self.slice(i, j)
			
			if win_flag:
				segment.ys *= window		# apply window function to the wave segment 

			t = (segment.start + segment.end) / 2		# the nominal time for this segment is the midpoint
			spectrum = segment.make_spectrum()
			spec_map[t] = spectrum

			i += step
			j += step

		return Spectrogram(spec_map, seg_length)

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

	def plot(self, title=None):
		# plots the wave

		time = np.linspace(self.start, self.duration, len(self.ys))	# get seconds

		pyplot.figure(figsize=(16,4))
		if title != None:
			pyplot.title(title)
		pyplot.plot(time, self.ys, color='#5F9EA0')
		pyplot.xlabel('Time(s)')
		pyplot.ylabel('Amplitude')
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
		# hs: array of amplitudes (real or complex)
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

	@property
	def power(self):
	    # return a sequence of powers (read-only property)

	    return self.amps**2
	

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

	def plot(self, title=None, high=None):
		# plots amplitude vs frequency
		#
		# if full spectrum, ignore low and high
		#
		# title: title of the plot
		# high: frequency to cut off at
	
		if self.full:
			fs, amps = self.render_full(high)

			pyplot.figure(figsize=(16,4))
			
			if title != None:
				pyplot.title(title)
			
			pyplot.plot(fs, amps, color='#5F9EA0')
			pyplot.xlabel('Frequency(Hz)')
			pyplot.ylabel('Amplitude')
			pyplot.show()
		else:
			i = None if high is None else find_index(high, self.fs)

			pyplot.figure(figsize=(16,4))
			
			if title != None:
				pyplot.title(title)
			
			pyplot.plot(self.fs[:i], self.amps[:i], color='#5F9EA0')
			pyplot.xlabel('Frequency(Hz)')
			pyplot.ylabel('Amplitude')
			pyplot.show()

	def plot_power(self, title=None, high=None):
		# plots power vs frequency
		#
		# title: title of the plot
		# high: frequency to cut off at

		if self.full:
			fs, amps = self.render_full(high)
			powers = amps**2

			pyplot.figure(figsize=(16,4))

			if title != None:
				pyplot.title(title)

			pyplot.plot(fs, powers, color='#5F9EA0')
			pyplot.xlabel('Frequency(Hz)')
			pyplot.ylabel('Power')
			pyplot.show()
		else:
			i = None if high is None else find_index(high, self.fs)

			pyplot.figure(figsize=(16,4))
			
			if title != None:
				pyplot.title(title)
			
			pyplot.plot(self.fs[:i], self.power[:i], color='#5F9EA0')
			pyplot.xlabel('Frequency(Hz)')
			pyplot.ylabel('Power')
			pyplot.show()

	def peaks(self):
		# find the highest amplitude peaks and their frequencies
		#
		# return: sorted list of (amplitude, frequency) pairs

		peaks = zip(self.amps, self.fs)
		peaks.sort(reverse=True)
		return peaks

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
		indices = (low_cutoff < fs) & (fs < high_cutoff)
		self.hs[indices] *= factor

	def freqToMel(self, freq):
	    # convert a frequency value to its Mel scale value

	    return 1127 * math.log(1 + freq / 700.0)

	def melToFreq(self, mel):
	    # convert a Mel scale value to its frequency value 

	    return 700 * (math.exp(mel / 1127.0 - 1))

	def melFilterBank(self, minHz, maxHz, numFilters, blocksize):
	    # compute the mel-spaced filterbank that will be applied to a power spectrum.
	    #
	    # minHz: lower frequency bound
	    # maxHz: upper frequency bound
	    # numFilters: the number of triangular filters (which is the number of coefficients)
	    # blocksize: the size of each filter (equal to (seg_length / 2) + 1)
	    #
	    # return: a matrix of triangular filters, each of length blocksize

	    minMel = self.freqToMel(minHz)
	    maxMel = self.freqToMel(maxHz)

	    filterMatrix = np.zeros((numFilters, blocksize))  # create matrix for triangular filters

	    # our range needs (numFilters + 2) linearly spaced points, each filter requires three points
	    melRange = np.array(xrange(numFilters + 2)) 

	    # calculate linearly spaced mel values between minMel and maxMel 
	    melCenterFilters = melRange * (maxMel - minMel) / (numFilters + 1) + minMel 

	    aux = np.log(1 + 1000.0 / 700.0) / 1000.0
	    aux = (np.exp(melCenterFilters * aux) - 1) / 22050
	    aux = 0.5 + 700 * blocksize * aux
	    aux = np.floor(aux)
	    centerIndex = np.array(aux, int)   # each index represents the center of each triangular filter

	    for i in xrange(numFilters):
	        start, center, end = centerIndex[i:i + 3]
	        k1 = np.float32(center - start)
	        k2 = np.float32(end - center)
	        up = (np.array(xrange(start, center)) - start) / k1
	        down = (end - np.array(xrange(center, end))) / k2
	        filterMatrix[i][start:center] = up
	        filterMatrix[i][center:end] = down

	    return filterMatrix.transpose()

	def get_mfcc(self, minHz=20, maxHz=22050, numFilters=26, blocksize=883):
		# compute the MFCCs of the given spectrum
		#
		# minHz: lower frequency bound
	    # maxHz: upper frequency bound
	    # numFilters: the number of triangular filters (which is the number of coefficients)
	    # blocksize: the size of each filter (equal to (segment_length / 2) + 1)
		# 
		# by default, assume the given spectrum has a segment_length of 1764 (which is 40ms at 44100fps)
		# 
		# return: feature vector (1D array of length numFilters)

		hs = self.hs    # length is equal to ((segment_length/2)+1) (result of FFT)
		fs = self.fs
		framerate = self.framerate
		full = self.full

		powers = self.power   # get spectrum power values

		# create the mel-filterbank, shape of the filterbank is (blocksize, numFilters)
		filterbank = self.melFilterBank(minHz, maxHz, numFilters, blocksize)  

		filtered_spectrum = np.dot(powers, filterbank)   # shape of filtered_spectrum is a 1D array of length numFilters

		log_spectrum = np.log(filtered_spectrum)   # take the logarithm of each of the filterbank energies 

		mfcc =  scipy.fftpack.dct(log_spectrum, type=2)  # take the DCT of these log filterbank energies

		return mfcc    

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

# TODO
class Dct(_SpectrumParent):
	# represents the spectrum of a signal using discrete cosine transform.

	@property
	def amps(self):
		# returns a sequence of amplitudes (read-only property)
		# NOTE: for DCT, amps are positive and negative real values

	    return self.hs

	def make_wave(self):
		# transforms to the time domain
		#
		# return: Wave

		n = len(self.hs)
		ys = scipy.fftpack.idct(self.hs, type=2) / 2 / n

		return Wave(ys, framerate=self.framerate)
	

#################################################################################################

class Spectrogram:
	# represents the spectrum of a signal over time

	def __init__(self, spec_map, seg_length):
		# initialize the spectrogram
		#
		# spec_map: map from float time to spectrum
		# seg_length: number of samples in each segment

		self.spec_map = spec_map
		self.seg_length = seg_length

	def times(self):
		# sorted sequence of times
		#
		# return: sequence of float times in seconds

		ts = sorted(iter(self.spec_map))
		return ts

	def any_spectrum(self):
		# returns an arbitrary spectrum from the spectrogram

		index = next(iter(self.spec_map))
		return self.spec_map[index]

	def frequencies(self):
		# sequence of frequencies
		#
		# return: sequence of float frequencies in Hz

		fs = self.any_spectrum().fs
		return fs

	def mfcc(self, minHz=20, maxHz=22050, numFilters=26, blocksize=883):
		# compute the MFCCs of the given set of spectrums
		#
		# minHz: lower frequency bound
	    # maxHz: upper frequency bound
	    # numFilters: the number of triangular filters (which is the number of coefficients per spectrum)
	    # blocksize: the size of each filter (equal to (segment_length / 2) + 1)
		# 
		# by default, assume each spectrum has a segment_length of 1764 (which is 40ms at 44100fps)
		# 
		# return: feature matrix (2D array of numFilters by number of spectrums (aka length of spec_map))

		mfcc_matrix = []
		for t, spectrum in sorted(self.spec_map.iteritems()):
			sub_mfcc = spectrum.get_mfcc(minHz, maxHz, numFilters, blocksize)   # get the mfcc of each spectrum in spec_map
			mfcc_matrix.append(sub_mfcc)

		mfcc_matrix = np.asarray(mfcc_matrix)    # typecast as numpy array

		return mfcc_matrix    # (number of spectrums, numFilters) feature matrix 



	def plot(self, title=None, high=None):
		# make a psuedocolor plot
		# 
		# high: highest frequency component to plot

		wave = self.make_wave()	

		dt = float(self.seg_length) / len(wave)
		# Fs = int(1.0/dt) * 10 * pow(2, math.log(self.seg_length, 2) - 8)		# the sampling frequency (samples per unit time)
		Fs = int(1.0/dt)
		NFFT = int(self.seg_length)
		noverlap = self.seg_length / 2			# the number of samples that each segment overlaps

		pyplot.figure(figsize=(16,4))

		if title != None:
			pyplot.title(title)

		# pyplot.xticks(np.arange(math.floor(wave.start), math.ceil(wave.end)+1, 1.0))
		pyplot.specgram(wave.ys, NFFT=NFFT, Fs=Fs, noverlap=noverlap, cmap=pyplot.cm.bone)
		pyplot.xlabel('Time(s)')
		pyplot.ylabel('Frequency(Hz)')
		pyplot.show()

	def make_wave(self):
		# inverts the spectrogram and returns a wave
		#
		# return: Wave

		res = []
		for t, spectrum in sorted(self.spec_map.iteritems()):
			wave = spectrum.make_wave()
			n = len(wave)

			window = 1 / np.hamming(n)
			wave.ys *= window

			i = wave.find_index(t)
			start = i - (n // 2)
			end = start + n
			res.append((start, end, wave))

		starts, ends, waves = zip(*res)
		low = min(starts)
		high = max(ends)

		ys = np.zeros(high - low, np.float)
		for start, end, wave in res:
			ys[start:end] = wave.ys

		return Wave(ys, framerate=wave.framerate)

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
