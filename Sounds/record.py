"""PyAudio example: Record a few seconds of audio and save to a WAVE file."""

import pyaudio
import wave

CHUNK = 1024					# fixed sample buffer size
FORMAT = pyaudio.paInt16		# 16-bit integer format
CHANNELS = 2					# 2 channels, left and right
RATE = 44100					# framerate, or samples per second
# RECORD_SECONDS = 5				# record up to 5 seconds
RECORD_SECONDS = 2.5				# record up to 2.5 seconds
WAVE_OUTPUT_FILENAME = "output.wav"

p = pyaudio.PyAudio()	# initialize PyAudio object

# open stream
stream = p.open(format=FORMAT,					
                channels=CHANNELS,
                rate=RATE,
                input=True,		# set stream to take input, or record		
                frames_per_buffer=CHUNK)

print("* recording")

frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("* done recording")

stream.stop_stream()		# stop stream
stream.close()				# close stream
p.terminate()				# terminate PyAudio object 

# save recording into output wav file
wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()