"""PyAudio Example: Play a WAVE file."""

import pyaudio
import wave
import sys

CHUNK = 1024		# fixed sample buffer size

# error message
if len(sys.argv) < 2:
    print("Plays a wave file.\n\nUsage: %s filename.wav" % sys.argv[0])
    sys.exit(-1)

wf = wave.open(sys.argv[1], 'rb')		# open wav file

p = pyaudio.PyAudio()		# initialize PyAudio object

# open stream
stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True)		# set stream to output

data = wf.readframes(CHUNK)

while data != '':
    stream.write(data)
    data = wf.readframes(CHUNK)

stream.stop_stream()		# stop stream
stream.close()				# close stream
p.terminate()				# terminate PyAudio object