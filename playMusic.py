#!

import pyaudio, wave
import numpy as np
from math import *
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


def play(file):
    # open the WAV file
    wf = wave.open(file, 'rb')
    # constants
    chunk = 2048
    RATE = wf.getframerate()
    # variables
    global thefreq
    thefreq = 1.0
    # open the stream
    p = pyaudio.PyAudio()
    stream = p.open(format=
                    p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=RATE,
                    output=True)
    # canvas to visualise
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    height = img.shape[0]
    width = img.shape[1]
    # read the incoming data
    data = wf.readframes(chunk)
    # play stream and find the frequency of each chunk
    while len(data) > 0:
        cv2.imshow('canvas', img)
        # write data out to the audio stream
        stream.write(data)
        data = wf.readframes(chunk)
        # unpack the data and turn into integer
        indata = np.frombuffer(data, np.int16)
        # Take the fft and square each value
        fftData = abs(np.fft.rfft(indata)) ** 2
        # find the maximum
        which = fftData[1:].argmax() + 1
        # use inverse quadratic interpolation to find the peak
        if which != len(fftData) - 1:
            y0, y1, y2 = np.log(fftData[which - 1:which + 2:])
            x1 = (y2 - y0) * .5 / (2 * y1 - y2 - y0)
            # find the frequency and output it
            thefreq = (which + x1) * RATE / chunk
            thefreq = which * RATE / chunk
            while thefreq < 350 and thefreq > 15:
                thefreq = thefreq * 2
            while thefreq > 700:
                thefreq = thefreq / 2

            c = 3 * 10 ** 8
            THz = thefreq * 2 ** 40
            pre = float(c) / float(THz)
            nm = int(pre * 10 ** (-floor(log10(pre))) * 100)
            rgb = wavelength_to_rgb(nm)
            cv2.line(img, (random.randrange(0, 500), random.randrange(0, 500)),
                     (random.randrange(0, 500), random.randrange(0, 500)), (rgb[2], rgb[1], rgb[0]), 10)
            print(rgb)
            k = cv2.waitKey(1) & 0xff
            if k == ord('q'):
                break
        # read some more data
        data = wf.readframes(chunk)
    if data:
        stream.write(data)
    stream.close()
    p.terminate()

play('lop13.wav')
