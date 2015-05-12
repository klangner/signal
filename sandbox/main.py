import cmath
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple

Wave = namedtuple("Wave", ['rate', 'data'])


def print_file_info(waveFileName: str):
    (rate, data) = wavfile.read(waveFileName)
    print(waveFileName)
    print("Rate: %d samples/s, time: %fs" % (rate, len(data)/rate))

def print_power(waveFileName: str):
    (rate, data) = wavfile.read(waveFileName)
    s = 0
    for x in data:
        s += x**2
    power = s/(2*len(data))
    print(waveFileName)
    print("Power %fs" % power)

def plot_wave(waveFileName: str):
    (rate, data) = wavfile.read(waveFileName)
    xaxis=np.linspace(0, len(data)/rate, num=len(data))
    plt.figure(1)
    plt.title(waveFileName)
    plt.plot(xaxis, data)
    plt.show()

def plot_sinusoid():
    N = 36
    omega = cmath.pi/6
    phi = 0 # cmath.pi/4
    signal = [cmath.exp(1j*(n*omega+phi)) for n in range(0, N)]
    plt.figure(1)
    plt.subplot(211)
    plt.title("Real part")
    plt.stem([x.real for x in signal])
    plt.subplot(212)
    plt.title("Imaginary")
    plt.stem([x.imag for x in signal], 'r')
    plt.show()


if __name__ == "__main__":
    print_power("../data/asa.wav")
    plot_sinusoid()

