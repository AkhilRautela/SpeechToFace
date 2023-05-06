from pathlib import Path
from statistics import mode
import numpy as np
import pandas as pd
import cv2
from keras.layers import *
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from helpers.audio_encoder import AudioFeatureExtraction

class Model():

    def __init__(self):
        self.audio_path = "data/audio/"
        self.extracted_face_path = "data/extracted_face/"
        self.dimension = (100, 100, 3)
        self.n_dim = 132300

    def create_dataset(self):
        audioFeatureExtractor = AudioFeatureExtraction(30)
        x_train = []
        y_train = []
        audioNameVsAudio = {}
        for file in os.listdir(self.audio_path):
            audioNameVsAudio[Path(file).resolve().stem] = audioFeatureExtractor.feature_extraction(file)
        for file in os.listdir(self.extracted_face_path):
            audio_fileName = Path(file).resolve().stem[:-1]
            if audio_fileName in audioNameVsAudio.keys():
                x_train.append(audioNameVsAudio.get(audio_fileName))
                y_train.append(cv2.imread(self.extracted_face_path + file))
        self.x_train = np.array(x_train)
        self.y_train = np.array(y_train)

    def create_generater_model(self):
        model = Sequential()
        model.add(Dense(625 , input_dim = self.n_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((25, 25, 1)))
        model.add(Conv2DTranspose(28, (3, 3), strides=(2,2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2DTranspose(32, (3, 3), strides=(2,2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(3, (100, 100), activation='sigmoid', padding='same'))
        model.summary()
        self.generator_model = model

    def create_discriminator_model(self):
        model = Sequential()
        model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same', input_shape=self.dimension))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        model.summary()
        self.disciminator_model = model
    
    def create_combined_model(self):
        model = Sequential()
        model.add(self.generator_model)
        model.add(self.disciminator_model)
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        model.summary()
        self.combined_model = model

    def train_model(self):
        self.combined_model.fit(self.x_train, )
        self.disciminator_model.train_on_batch()

    def train_generator_model(self):
        opt = Adam(lr=0.0002, beta_1=0.5)
        self.generator_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        print(self.x_train.shape)
        print(self.y_train.shape)
        self.generator_model.fit(self.x_train, self.y_train, batch_size=1, epochs = 10, verbose = True)

    def save_model(self):
        pass

    def load_model(self):
        pass

if __name__ == "__main__":
    model = Model()
    model.create_dataset()
    model.create_generater_model()
    model.create_discriminator_model()
    model.create_combined_model()
    model.train_generator_model()