"""PyAudio example: Record audio and save interval samples to WAVE files."""

import pyaudio
import wave

CHUNK = 1024					# fixed sample buffer size
FORMAT = pyaudio.paInt16		# 16-bit integer format
CHANNELS = 2					# 2 channels, left and right
RATE = 44100					# framerate, or samples per second
RECORD_SECONDS = 2.5				# record up to 2.5 seconds
WAVE_OUTPUT_FILENAME = "Output/output"

p = pyaudio.PyAudio()	# initialize PyAudio object

# open stream
stream = p.open(format=FORMAT,					
                channels=CHANNELS,
                rate=RATE,
                input=True,		# set stream to take input, or record		
                frames_per_buffer=CHUNK)

print("* recording")

count = 0
while (True):	
	frames = []

	for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
	    data = stream.read(CHUNK)
	    frames.append(data)

	# save recording into output wav file
	wf = wave.open(WAVE_OUTPUT_FILENAME+str(count)+".wav", 'wb')
	wf.setnchannels(CHANNELS)
	wf.setsampwidth(p.get_sample_size(FORMAT))
	wf.setframerate(RATE)
	wf.writeframes(b''.join(frames))
	wf.close()

	count += 1

stream.stop_stream()		# stop stream
stream.close()				# close stream
p.terminate()				# terminate PyAudio object 

print("* done recording")