from fileinput import filename
import librosa
import numpy as np

class AudioFeatureExtraction():

    def __init__(self, feature_count):
        self.feature_count = feature_count
        self.audio_path = "data/audio/"

    def feature_extraction(self, filename):
        aud, sr = librosa.load(self.audio_path + filename)
        return librosa.feature.mfcc(aud, sr, self.feature_count)

