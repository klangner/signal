
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np


def printFileInfo(waveFileName: str):
    (rate, data) = wavfile.read(waveFileName)
    print(waveFileName)
    print("Rate: %d samples/s, time: %fs" % (rate, len(data)/rate))


def plotWave(waveFileName: str):
    (rate, data) = wavfile.read(waveFileName)
    xaxis=np.linspace(0, len(data)/rate, num=len(data))
    plt.figure(1)
    plt.title(waveFileName)
    plt.plot(xaxis, data)
    plt.show()

if __name__ == "__main__":
    printFileInfo("../data/asa.wav")
