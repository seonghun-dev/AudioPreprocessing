# -*- coding: utf-8 -*-
import argparse
import os

import numpy as np
import random
import soundfile as sf
from enum import Enum
import re

import scipy.signal as signal
import math
import librosa
import matplotlib.pyplot as plt
import librosa.display
from PIL import Image

def wav_fft(file_name):
    audio_sample, sampling_rate = librosa.load(file_name, sr = None)
    fft_result = librosa.stft(audio_sample, n_fft=1024, hop_length=512, win_length = 1024, window=signal.hann).T
    mag, phase = librosa.magphase(fft_result)
    return mag

#normalize_function
min_level_db = -100
def _normalize(S):
    return np.clip((S-min_level_db)/(-min_level_db), 0, 1)

class EncodingType(Enum):
    def __new__(cls, *args, **kwds):
        value = len(cls.__members__) + 4
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, dtype, description, subtype, maximum, minimum):
        self.dtype = dtype
        self.description = description
        self.subtype = subtype
        self.maximum = maximum
        self.minimum = minimum

    # Available subtypes
    # See. https://pysoundfile.readthedocs.io/en/latest/#soundfile.available_subtypes
    INT16 = (
        "int16",
        "Signed 16 bit PCM",
        "PCM_16",
        np.iinfo(np.int16).max,
        np.iinfo(np.int16).min,
    )
    INT32 = (
        "int32",
        "Signed 32 bit PCM",
        "PCM_32",
        np.iinfo(np.int32).max,
        np.iinfo(np.int32).min,
    )
    FLOAT32 = ("float32", "32 bit float", "FLOAT", 1, -1)
    FLOAT64 = ("float64", "64 bit float", "DOUBLE", 1, -1)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_clean_file", type=str, default="")
    parser.add_argument("--output_noise_file", type=str, default="")
    args = parser.parse_args()
    return args


def cal_adjusted_rms(clean_rms, snr):
    a = float(snr) / 20
    noise_rms = clean_rms / (10 ** a)
    return noise_rms


def cal_rms(amp):
    return np.sqrt(np.mean(np.square(amp), axis=-1))


def save_waveform(output_path, amp, samplerate, subtype):
    sf.write(output_path, amp, samplerate, format="wav", subtype=subtype)


if __name__ == "__main__":
    args = get_args()
    clean_dir = './Original'
    noise_dir = "./Noise"
    clean_files = os.listdir(clean_dir)
    noise_files = os.listdir(noise_dir)
    print(clean_files[0])
    for noise_file_name in noise_files :
        for clean_file_name in clean_files :
            clean_file = "./Original/" + clean_file_name
            noise_file = "./Noise/" +noise_file_name
            clean_file_source = re.findall('(?<=[l][/]).*(?=[.][w][a][v])',clean_file)
            noise_file_source = re.findall('(?<=[e][/]).*(?=[.][w][a][v])',noise_file)
            for i in range(100):
                snr = float(i+1)
                outputfile_name = clean_file_source[0] +"." + noise_file_source[0] +"."+"snr"+ str(int(snr))
                print("Now make "+outputfile_name)
                output_file = "./Output/"+outputfile_name+".wav"
                output_file_PNG = "./Output_PNG/" + outputfile_name + ".png"

                metadata = sf.info(clean_file)
                for item in EncodingType:
                    if item.description == metadata.subtype_info:
                        encoding_type = item

                clean_amp, clean_samplerate = sf.read(clean_file, dtype=encoding_type.dtype)
                noise_amp, noise_samplerate = sf.read(noise_file, dtype=encoding_type.dtype)

                clean_rms = cal_rms(clean_amp)

                start = random.randint(0, len(noise_amp) - len(clean_amp))
                divided_noise_amp = noise_amp[start : start + len(clean_amp)]
                noise_rms = cal_rms(divided_noise_amp)


                adjusted_noise_rms = cal_adjusted_rms(clean_rms, snr)

                adjusted_noise_amp = divided_noise_amp * (adjusted_noise_rms / noise_rms)
                mixed_amp = clean_amp + adjusted_noise_amp

                # Avoid clipping noise
                max_limit = encoding_type.maximum
                min_limit = encoding_type.minimum
                if mixed_amp.max(axis=0) > max_limit or mixed_amp.min(axis=0) < min_limit:
                    if mixed_amp.max(axis=0) >= abs(mixed_amp.min(axis=0)):
                        reduction_rate = max_limit / mixed_amp.max(axis=0)
                    else:
                        reduction_rate = min_limit / mixed_amp.min(axis=0)
                    mixed_amp = mixed_amp * (reduction_rate)
                    clean_amp = clean_amp * (reduction_rate)

                save_waveform(
                    output_file, mixed_amp, clean_samplerate, encoding_type.subtype
                )

                mag = wav_fft(output_file)
                mag_db = librosa.amplitude_to_db(mag)
                mag_n = _normalize(mag_db)

                librosa.display.specshow(mag_n.T, y_axis='linear', x_axis='time', sr=16000)
                plt.title('FFT result')
                plt.savefig(output_file_PNG)