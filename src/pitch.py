import numpy as np
from scipy.signal import decimate

from src.utils import read_fft_data, parabolic


def find_peaks(wave_file_name: str):
    print(wave_file_name)
    (rate, xs) = read_fft_data(wave_file_name)
    res = rate/4096.0
    peaks = find_spectra_peaks(xs)
    for peak in peaks:
        print("Freq: %dHz, Value: %f" % ((res*peak), np.log10(xs[peak])))


def find_spectra_peaks(spectrum):
    peaks = []
    m = np.mean(spectrum)
    for i in range(9, len(spectrum)-1):
        if spectrum[i-1] < spectrum[i] and spectrum[i] > spectrum[i+1] and spectrum[i] > m:
            peaks.append(parabolic(spectrum, i)[0])
    return peaks


def freq_from_hps(wave_file_name: str):
    """ Estimate frequency using harmonic product spectrum
    """
    print(wave_file_name)
    (rate, xs) = read_fft_data(wave_file_name)[:500]
    N = 2*len(xs)
    peak = hps_pitch_detector(xs)
    print("F0 = %f" % ((rate * peak)/N))


def hps_pitch_detector(spectrum):
    """ Estimate frequency using harmonic product spectrum
    """
    N = 2*len(spectrum)
    hps = np.copy(spectrum)
    dec = []
    for h in range(2, 5):
        dec = decimate(spectrum, h)
        hps[:len(dec)] *= dec
    # Find the peak and interpolate to get a more accurate peak
    i_peak = np.argmax(hps[:len(dec)])
    return parabolic(hps, i_peak)[0]


def freq_from_hps_ex(wave_file_name: str):
    print(wave_file_name)
    (rate, xs) = read_fft_data(wave_file_name)[:500]
    N = 2*len(xs)
    hps_peak = hps_pitch_detector(xs)
    spectra_peaks = find_spectra_peaks(xs)
    peak = hps_peak
    distance = len(xs)
    for p in spectra_peaks:
        if abs(p-hps_peak) < distance:
            distance = abs(p-hps_peak)
            peak = p
    print("HPS with correction F0 = %f" % ((rate * peak)/N))


if __name__ == "__main__":
    find_peaks("data/a4.wav")
    # freq_from_hps("data/440Hz.wav")
    # freq_from_hps_ex("data/440Hz.wav")
    freq_from_hps_ex("data/a4.wav")

