#!/usr/bin/python3

import sys
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
import librosa
import essentia

from key_detector import Key_Detector

def main(path):

	detector = Key_Detector(path)
	print('BPM: ' + str(detector.GetBPM()))
	detector.GetSongKey()
	#detector.ComputeKeyOnTimeLine()

if __name__ == '__main__':

	path = None

	for arg in sys.argv:
		if arg == '-f':
			try:
				path = sys.argv[sys.argv.index(arg)+1]
			except IndexError:
				print("Path was not given")
				sys.exit(1)
	
	if path == None:
		print("Path was not given")
		sys.exit(1)

	main(path)