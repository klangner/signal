import math

from scipy.io import wavfile
import numpy as np


def print_power(data):
    s = 0
    for x in data:
        s += pow(x, 2.0)
    power = math.log10(math.sqrt(s))/math.log10(pow(2, 31))
    print("Power %d%%" % (power*100))


def print_note_filter_bank(waveFileName: str):
    (rate, data) = wavfile.read(waveFileName)
    # Take samples from the beginning of the signal
    xs = data[:4096]
    fft = np.fft.fft(xs)
    fb = filter_bank(rate, fft, 12, 220, 440)
    # Normalize
    mx = np.max(fb)
    fb = fb / mx
    print(fb)


def filter_bank(rate, data, bins = 10, start_freq=300, end_freq=5000):
    '''
    :param data: DFT spectrum
    :param bins: how many bins?
    :return: list with freq powers
    '''
    resolution = rate/len(data)
    start = int(start_freq/resolution)
    end = int(end_freq/resolution)
    bank = np.zeros(bins)
    for i in range(start, end):
        x = np.absolute(data[i])
        pos = np.floor((i-start)/resolution)
        if pos < bins:
            bank[pos] += x
    return bank


if __name__ == "__main__":
    print_note_filter_bank("../data/e6.wav")

