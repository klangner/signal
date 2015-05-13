import cmath
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
import math
from collections import namedtuple

Wave = namedtuple("Wave", ['rate', 'data'])


def print_file_info(waveFileName: str):
    (rate, data) = wavfile.read(waveFileName)
    print(waveFileName)
    print("Rate: %d samples/s, time: %fs" % (rate, len(data)/rate))


def print_power(data):
    s = 0
    for x in data:
        s += pow(x, 2.0)
    power = math.log10(math.sqrt(s))/math.log10(pow(2, 31))
    print("Power %d%%" % (power*100))


def plot_wave(waveFileName: str):
    (rate, data) = wavfile.read(waveFileName)
    print_power(data)
    xaxis=np.linspace(0, len(data)/rate, num=len(data))
    plt.figure(1)
    plt.title(waveFileName)
    plt.plot(xaxis, data)
    plt.show()


def plot_sinusoid():
    N = 36
    omega = cmath.pi/6
    phi = cmath.pi/4
    x = [cmath.exp(1j*(n*omega+phi)).real for n in range(0, N)]
    plt.figure(1)
    plt.title("Re cos(n*pi/6 + pi/4)")
    plt.plot(x)
    plt.ylim([-20000, 20000])
    plt.show()


def plot_freq(waveFileName: str):
    (rate, data) = wavfile.read(waveFileName)
    xs = data[:4096]*np.hanning(4096)
    fft = np.fft.fft(xs)
    plt.figure(1)
    plt.title("FFT")
    xs = np.absolute(fft[:100])
    plt.plot(xs)
    plt.show()


def find_peaks(waveFileName: str):
    print(waveFileName)
    (rate, data) = wavfile.read(waveFileName)
    start = int(len(data)/2)
    xs = data[start:start+4096]*np.hanning(4096)
    fft = np.fft.fft(xs)
    xs = np.absolute(fft[:500])
    peaks = []
    res = rate/4096.0
    m = np.mean(xs)
    for i in range(9, len(xs)-1):
        if xs[i-1] < xs[i] and xs[i] > xs[i+1] and xs[i] > m:
            peak = i + peak_interpolation(xs[i-1], xs[i], xs[i+1])
            peaks.append(peak)
            print("Freq: %dHz, Value: %f" % ((res*peak), np.log10(xs[i])))


def peak_interpolation(a, b, c):
    return ((a-c)/(a-2*b+c))/2


def print_filter_bank(waveFileName: str, bins=10):
    (rate, data) = wavfile.read(waveFileName)
    # Take samples from the beginning of the signal
    xs = data[:4096]
    fft = np.fft.fft(xs)
    fb = filter_bank(rate, fft, bins)
    # Normalize
    mx = np.max(fb)
    fb = fb / mx
    print(fb)


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
    # plot_wave("../data/100Hz.wav")
    # plot_freq("../data/100Hz.wav")
    find_peaks("../data/asa.wav")
    # plot_wave("../data/a4.wav")
    # print_filter_bank("../data/e6.wav")
    # print_note_filter_bank("../data/e6.wav")

