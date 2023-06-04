#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
import librosa
import librosa.display
import essentia.standard as es
import aubio
from numpy import median, diff
import scipy

class Key_Detector(object):

    def __init__(self, path, time_start = None, time_end = None, method='essentia'):

        self.path_of_song = path
        self.__time_start = time_start
        self.__time_end = time_end
        self.method = method
        self.song_info = None
        self.audio = None
        self.key_song = None
        self.__keys = []
        self.__bpm = None
        self.__chroma_values = []
        self.__LoadAudioFile()
        self.__InitTimeLine()
        self.__Init_bpm()
        self.__Init_Chromagram()

    def __Init_bpm(self):

        if self.method == 'essentia':
            self.__EssentiaMethodBPM()
        elif self.method == 'aubio':
            self.__AubioMethodBPM()
        elif self.method == 'librosa':
            self.__LibrosaMethodBPM()
        else:
            self.method = 'essentia'
            print("BPM getting Method was changed to default value: {0}".format("essentia"))
            self.__EssentiaMethodBPM()

    def __Init_Chromagram(self, offset = 0.0, duration = None):
        
        y, sr = librosa.load(self.path_of_song, offset = offset, duration = duration)
        self.__chroma_values = librosa.feature.chroma_stft(y=y,sr=sr)
        print('Chroma'+'\n')
        print(len(self.__chroma_values[1]))


    def __LoadAudioFile(self):

        self.audio = es.MonoLoader(filename=self.path_of_song)()

    def __InitTimeLine(self):

        if self.__time_start == None or self.__time_end == None:
            self.__time_start = 0
            duration_extractor = es.Duration()
            self.__time_end = int(duration_extractor(self.audio) // 1)

    def __EssentiaMethodBPM(self):

        self.__bpm = es.PercivalBpmEstimator()(self.audio[self.__time_start*44100:self.__time_end*44100])

    def __AubioMethodBPM(self):

        source = aubio.source(self.path_of_song)
        tempo = aubio.tempo(method="default", buf_size=2048, hop_size=source.hop_size, samplerate=source.samplerate)
        beats = []

        total_frames = 0

        while True:
            samples, read = source()
            is_beat = tempo(samples)
            if is_beat:
                this_beat = tempo.get_last_s()
                beats.append(this_beat)
                #if o.get_confidence() > .2 and len(beats) > 2.:
                #    break
            total_frames += read
            if read < source.hop_size:
                break

        if len(beats) > 1:
            if len(beats) < 4:
                print("few beats found in {:s}".format(path))
            bpms = 60./diff(beats)
            self.__bpm = median(bpms)
        else:
            self.__bpm = 0
            print("not enough beats found in {:s}".format(path))

    def __LibrosaMethodBPM(self):

        pass

    def __Key_Finding_Algorithm(self, method = 'Krumhansl-Schmuckler'): 

        pitches = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
        maj_profile = []
        min_profile = []
        #profiles for correlation
        if (method == 'Krumhansl-Schmuckler'):
            maj_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
            min_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
        elif (method == 'Aarden-Essen'):
            maj_profile = [17.7661,14.9265,0.145624,0.160186,19.8049,11.3587,0.291248,22.062,0.145624,8.15494,0.232998,4.95122]
            min_profile = [18.2648,0.737619,14.0499,16.8599,0.702494,14.4362,0.702494,18.6161,4.56621,1.93186,7.37619,1.75623]
        elif (method == 'Simple'):
            maj_profile = [2,0,1,0,1,1,0,2,0,1,0,1]
            min_profile = [2,0,1,1,0,1,0,2,1,0,0.5,0.5]
        elif (method == 'Bellman-Budge'):
            maj_profile = [16.8,0.86,12.95,1.41,13.49,11.93,1.25,20.28,1.8,8.04,0.62,10.57]
            min_profile = [18.16,0.69,12.99,13.34,1.07,11.15,1.38,21.07,7.49,1.53,0.92,10.21]
        elif (method == 'Temperley-Kostka-Payne'):
            maj_profile = [0.748,0.06,0.488,0.082,0.67,0.46,0.096,0.715,0.104,0.366,0.057,0.4]
            min_profile = [0.712,0.084,0.474,0.618,0.049,0.46,0.105,0.747,0.404,0.067,0.133,0.33]

        maj_profile = scipy.stats.zscore(maj_profile)
        min_profile = scipy.stats.zscore(min_profile)
        print('After Z-scrore: ' + '\n')
        print(maj_profile)
        print(min_profile)
        print('\n')
        maj_norm = scipy.linalg.norm(maj_profile)
        min_norm = scipy.linalg.norm(min_profile)
        print('After norm: ' + '\n')
        print(maj_profile)
        print(min_profile)
        print('\n')
        maj_profile = scipy.linalg.circulant(maj_profile)
        min_profile = scipy.linalg.circulant(min_profile)
        print('After circulant: ' + '\n')
        print(maj_profile)
        print(min_profile)
        print('\n')

        pitch_class_dist = self.__chroma_values.sum(axis=1)

        pitch_class_dist = scipy.stats.zscore(pitch_class_dist)
        pitch_class_dist_norm = scipy.linalg.norm(pitch_class_dist)

        self.coeffs_major = maj_profile.T.dot(pitch_class_dist)/(maj_norm*pitch_class_dist_norm)
        self.coeffs_minor = min_profile.T.dot(pitch_class_dist)/(min_norm*pitch_class_dist_norm)

        if (np.max(self.coeffs_major) > np.max(self.coeffs_minor)):
            self.key_song = str(pitches[np.argmax(self.coeffs_major)]) + ' major'
            self.coeff_correlation = np.max(self.coeffs_major)
        else:
            self.key_song = str(pitches[np.argmax(self.coeffs_minor)]) + ' minor'
            self.coeff_correlation = np.max(self.coeffs_minor)

        self.__keys.append([self.key_song, method])

    def __FormCoeffTable(self):

        pitches = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

        for i, coeff in enumerate(self.coeffs_major):
            print("{0} major: {1}".format(pitches[i], coeff))

        for i, coeff in enumerate(self.coeffs_minor):
            print("{0} minor: {1}".format(pitches[i], coeff))

    def ComputeKeyOnTimeLine(self):

        prev_key = self.key_song
        prev_coeff = self.coeff_correlation
        time_start = self.__time_start
        for i in range(self.__time_end):
            #print(i+1)
            self.__Init_Chromagram(self.__time_start, i+1)
            self.__Krumhansl_Schmuckler_Algorithm()
            print("In timeline from {0} to {1} Key of song is {2} (coeffiecient: {3})".format(time_start, self.__time_start+i+1, prev_key, prev_coeff))
            if (self.key_song != prev_key):
                prev_key = self.key_song
                prev_coeff = self.coeff_correlation
                time_start = self.__time_start + i+1

    def __PrintKey(self):

        print("likely key: ", self.key_song, ", correlation: ", self.coeff_correlation, sep='')

    def __GetMostExpectedKey(self):

        pitches = {'C major':0,'C# major':0,'D major':0,'D# major':0,'E major':0,'F major':0,'F# major':0,'G major':0,'G# major':0,'A major':0,'A# major':0,'B major':0, 'C minor':0,'C# minor':0,'D minor':0,'D# minor':0,'E minor':0,'F minor':0,'F# minor':0,'G minor':0,'G# minor':0,'A minor':0,'A# minor':0,'B minor':0}
        for pair in self.__keys:
            pitches[pair[0]] += 1

        print("Most expected Key of song: {0}".format(max(pitches, key = pitches.get)))


    def GetSongKey(self):

        print('\n')
        print('Krumhansl-Schmuckler Algorithm: ')
        self.__Key_Finding_Algorithm()
        self.__PrintKey()
        print('\n')
        print('Aarden-Essen Algorithm: ')
        self.__Key_Finding_Algorithm('Aarden-Essen')
        self.__PrintKey()
        print('\n')
        print('Simple Algorithm: ')
        self.__Key_Finding_Algorithm('Simple')
        self.__PrintKey()
        print('\n')
        print('Bellman-Budge Algorithm: ')
        self.__Key_Finding_Algorithm('Bellman-Budge')
        self.__PrintKey()
        print('\n')
        print('Temperley-Kostka-Payne Algorithm: ')
        self.__Key_Finding_Algorithm('Temperley-Kostka-Payne')
        self.__PrintKey()
        #self.__FormCoeffTable()
        print('\n')
        self.__GetMostExpectedKey()

    def GetBPM(self):

        return self.__bpm