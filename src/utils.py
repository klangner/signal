import cmath
from collections import namedtuple

from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np

Wave = namedtuple("Wave", ['rate', 'data'])


def print_file_info(waveFileName: str):
    (rate, data) = wavfile.read(waveFileName)
    print(waveFileName)
    print("Rate: %d samples/s, time: %fs" % (rate, len(data)/rate))


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


def plot_fft(fft):
    plt.figure(1)
    xs = np.absolute(fft[:100])
    plt.plot(xs)
    plt.show()


def parabolic(f, x):
    """Quadratic interpolation for estimating the true position of an
    inter-sample maximum when nearby samples are known.
    f is a vector and x is an index for that vector.
    Returns (vx, vy), the coordinates of the vertex of a parabola that goes
    through point x and its two neighbors.
    Example:
    Defining a vector f with a local maximum at index 3 (= 6), find local
    maximum if points 2, 3, and 4 actually defined a parabola.
    In [3]: f = [2, 3, 1, 6, 4, 2, 3, 1]
    In [4]: parabolic(f, argmax(f))
    Out[4]: (3.2142857142857144, 6.1607142857142856)
    """
    xv = 1/2. * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
    yv = f[x] - 1/4. * (f[x-1] - f[x+1]) * (xv - x)
    return xv, yv



def read_fft_data(wave_file_name: str):
    """
    :param wave_file_name:
    :return: first half of the spectrum
    """
    (rate, data) = wavfile.read(wave_file_name)
    start = int(len(data)/2)
    xs = data[start:start+4096]*np.hanning(4096)
    fft = np.fft.fft(xs)
    return rate, np.absolute(fft[:len(xs)/2])
