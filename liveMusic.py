import pyaudio
import matplotlib.pyplot as plt
import numpy as np
import math
# fourier transform
from scipy.fftpack import fft
import cv2
import random


def wavelength_to_rgb(wavelength, gamma=0.8):
    '''This converts a given wavelength of light to an
    approximate RGB color value. The wavelength must be given
    in nanometers in the range from 380 nm through 750 nm
    (789 THz through 400 THz).

    Based on code by Dan Bruton
    http://www.physics.sfasu.edu/astro/color/spectra.html
    '''

    wavelength = float(wavelength)
    if wavelength >= 380 and wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif wavelength >= 440 and wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif wavelength >= 490 and wavelength <= 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif wavelength >= 510 and wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif wavelength >= 580 and wavelength <= 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif wavelength >= 645 and wavelength <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0
    R *= 255
    G *= 255
    B *= 255
    return (int(R), int(G), int(B))


def live_singing():
    # constants
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 1024 * 2
    # variables
    global frequency
    # PyAudio object
    mic = pyaudio.PyAudio()
    # Record from mic
    stream = mic.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, output=True, frames_per_buffer=CHUNK)
    # matplotlib figure and axes
    fig, (ax, ax2) = plt.subplots(2, figsize=(15, 7))
    x = np.arange(0, 2 * CHUNK, 2)
    x_fft = np.linspace(0, RATE, CHUNK)
    # create line object
    line, = ax.plot(x, np.random.rand(CHUNK))
    line_fft, = ax2.semilogx(x_fft, np.random.rand(CHUNK))
    # axes formatting
    ax.set_ylim(-255, 255)  # symmetrical to the x axis
    ax.set_xlim(0, CHUNK)  # make sure our x axis matched our chunk size
    ax2.set_xlim(20, RATE / 2)

    # canvas to draw
    img = np.ones((500, 500, 3), dtype=np.uint8)
    while True:
        cv2.imshow("img", img)
        data = stream.read(CHUNK)
        data_int = np.frombuffer(data, np.int16)
        # graphing on plt
        line.set_ydata(data_int)
        y_fft = fft(data_int)
        line_fft.set_ydata(np.abs(y_fft[0:CHUNK]) * 2 / (256 * CHUNK))
        # spectrum of sound
        spectrum = np.abs(y_fft[0:CHUNK]) * 2 / (256 * CHUNK)
        # find peak
        peak = np.where(spectrum > 0.5)
        frequency_arr = [x_fft[i] for i in peak]
        frequency_arr = np.array(frequency_arr)
        # clean up noises
        epsilon = 200
        frequency_clean = np.delete(frequency_arr, np.where(frequency_arr < 380 - epsilon))
        frequency_clean = np.delete(frequency_arr, np.where(frequency_arr > 780 + epsilon))
        frequency = np.mean(frequency_clean)
        if not math.isnan(frequency):
            c = 3 * 10 ** 8  # speed of light
            THz = frequency * 2 ** 40  # convert sound fre to light fre
            if THz != 0:
                wavelen = float(c) / float(THz)
            nm = int(wavelen * 10 ** (-math.floor(math.log10(wavelen))) * 100)
            rgb = wavelength_to_rgb(nm)
            cv2.line(img, (random.randrange(0, 500), random.randrange(0, 500)),
                     (random.randrange(0, 500), random.randrange(0, 500)), (rgb[2], rgb[1], rgb[0]), 10)
            print(str(rgb))
        k = cv2.waitKey(1) & 0xff
        if k == ord('q'):
            break
        # display output
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.01)
    stream.close()
    mic.terminate()
live_singing()
