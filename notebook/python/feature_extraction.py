#!/usr/bin/env python
# coding: utf-8

# ## About
# visualize and analyse audio features
# 
# ## Inputs
# (4) audio files: (2) kpop audio files, (2) lofi audio files     
# KPOP:
# 1. LIKEY_TWICE.wav  
# 2. dynamite_bts.wav  
# 
# LOFI:  
# 1. 2019-06-25_-_Bobbin_-_www.FesliyanStudios.com_-_David_Renda.wav
# 2. Ambivert by _ e s c p _ [ Electronica _ Lo-Fi _ Synthwave _ Chillwave ] _ free-stock-music.com.wav 
#    
# ## Background
# 
# MIR System Architecture:  
# Audio --> [Segmentation] --> [Feature Extraction] --> [Machine Learning] --> Music Information
# 
# ## Resources
# https://github.com/stevetjoa/musicinformationretrieval.com  
# https://www.youtube.com/watch?v=oGGVvTgHMHw  
# https://librosa.org/doc/latest/feature.html  
# https://www.mathworks.com/help/audio/ug/spectral-descriptors.html#SpectralDescriptorsExample-6  
# https://musicinformationretrieval.com/spectral_features.html  
# https://towardsdatascience.com/how-i-understood-what-features-to-consider-while-training-audio-files-eedfb6e9002b
# 
# 

# In[ ]:


import numpy as np
import librosa
import librosa, librosa.display
import matplotlib.pyplot as plt
import IPython.display as ipd
from IPython.display import Audio, display

from pathlib import Path
from random import choices, randint
import mir_eval
import scipy
import sklearn


# ## Temporal Features
# The temporal features (time domain features), which are simple to extract and have easy physical interpretation, like: the energy of signal, zero crossing rate, maximum amplitude, minimum energy, etc.
# 
# ### Zero cross rate
# Often used to classify percussive sounds. A large number of zero crossings implies that there is no dominant low-frequency oscillation. 

# In[ ]:


def feature_extraction(data, srate, hop_size=512 ):
    '''
    Extracts the features
    
    This will be used to organize the different sounds in lofi music (drums, bass, high_notes, etc)
    
    params:
        data: the audio data array of a single audio clip to extract features from
        srate: sample rate
        hop_length: hop length
        
    returns:
        an array of features
            [0]: zero cross rate mean
            [1]: zero cross rate standard deviation
            [2]: spectral centroid mean
            [3]: spectral centroid standard deviation
            [4]: spectral bandwidth mean
            [5]: spectral bandwidth standard deviation
            [6]: spectral flatness mean
            [7]: spectral flatness standard deviation
            [8]: spectral rolloff mean
            [9]: spectral rolloff standard deviation
    '''
    # Features
    zcr = librosa.zero_crossings(data)
    
    features = [zcr.mean(), zcr.std()]
    
    #SPECTRAL FEATURES
    spectral_features = extract_spectral(data, srate, hop_size)  #store spectral features in array
    
    #we may need to normalize these features. See Scikit,learn MinMaxScaler
    
    return np.concatenate([features, spectral_features])


# ## Spectral Features
# The spectral features (frequency based features), which are obtained by converting the time based signal into the frequency domain using the Fourier Transform, like: mfcc, fundamental frequency, frequency components, spectral centroid, spectral flux, spectral density, spectral roll-off, etc. **These features can be used to identify the notes, pitch, rhythm, and melody.**  
# 
# ### spectral centroid
# The [spectral centroid](https://en.wikipedia.org/wiki/Spectral_centroid#:~:text=The%20spectral%20centroid%20is%20a,of%20brightness%20of%20a%20sound.) is a measure used in digital signal processing to characterise a spectrum. It indicates where the center of mass of the spectrum is located. Perceptually, it has a robust connection with the impression of **brightness** of a sound. Because the spectral centroid is a good predictor of the "brightness" of a sound, it is widely used in digital audio and music processing as an automatic measure of musical timbre.
# 
# ### spectral bandwidth
# Bandwidth is the difference between the upper and lower frequencies in a continuous band of frequencies. It is typically measured in hertz, and depending on context, may specifically refer to passband bandwidth or baseband bandwidth. Passband bandwidth is the difference between the upper and lower cutoff frequencies of, for example, a band-pass filter, a communication channel, or a signal spectrum. Baseband bandwidth applies to a low-pass filter or baseband signal; the bandwidth is equal to its upper cutoff frequency.
# 
# ### spectral flatness 
# Spectral flatness is an indication of the peakiness of the spectrum. Spectral flatness is typically measured in decibels, and provides a way to quantify **how tone-like a sound is, as opposed to being noise-like.** A higher spectral flatness indicates noise, while a lower spectral flatness indicates tonality. Tonality, in music, principle of organizing musical compositions around a central note, the tonic. Generally, any Western or non-Western music periodically returning to a central, or focal, tone exhibits tonality
# 
# ### spectral rolloff
# This is a measure of the amount of the right-skewedness of the power spectrum. The spectral roll off point is the fraction of bins in the power spectrum at which 85% of the power is at lower frequencies. That is, the roll-off is the frequency below which 85% of accumulated spectral magnitude is concentrated. Like the centroid, it takes on higher values for right-skewed spectra.
# 
# The spectral rolloff point has been used to distinguish between voiced and unvoiced speech, speech/music discrimination, music genre classification, acoustic scene recognition, and music mood classification. 

# In[ ]:


# Taken from Jordie's 'Audio Feature Extraction' notebook
# called in 'feature_extraction(data)'
def extract_spectral(data, srate, hop_length=512):
    '''
    Extracts spectral features
    
    Called in 'feature_extraction(data)'
    
    Taken from Jordie's 'Audio Feature Extraction' notebook

    params:
        data: the audio data array of a single audio clip to extract features from
        srate: sample rate
        hop_length: hop length
        
    returns:
        an list of np arrays called a feature vector
    '''
    # np.array
    centroid = librosa.feature.spectral_centroid(data+0.01, sr=srate, hop_length=hop_length) #computes the spectral centroid for each frame in a signal
    bandwidth = librosa.feature.spectral_bandwidth(data, sr=srate, hop_length=hop_length)
    flatness = librosa.feature.spectral_flatness(data, hop_length=hop_length)
    rolloff = librosa.feature.spectral_rolloff(data, sr=srate, hop_length=hop_length)
    
    '''Similar to the zero crossing rate, there is a spurious rise in spectral centroid 
    at the beginning of the signal. That is because the silence at the beginning has 
    such small amplitude that high frequency components have a chance to dominate. 
    One hack around this is to add a small constant before computing the spectral 
    centroid, thus shifting the centroid toward zero at quiet portions:'''
    
    # feature vector list
    feature_vector = [
        centroid.mean(),
        centroid.std(),
        bandwidth.mean(),
        bandwidth.std(),
        flatness.mean(),
        flatness.std(),
        rolloff.mean(),
        rolloff.std()
    ]
    
    return feature_vector


# In[ ]:


#load audio samples
# Default librosa sample rate
# sr = 22050
sr = 44100

kpop_in1, _ = librosa.load("LIKEY_TWICE.wav", sr=sr)
kpop_in2, _ = librosa.load("dynamite_bts.wav", sr=sr)
lofi_in1, _ = librosa.load("2019-06-25_-_Bobbin_-_www.FesliyanStudios.com_-_David_Renda.wav", sr=sr)
lofi_in2, _ = librosa.load("Ambivert by _ e s c p _ [ Electronica _ Lo-Fi _ Synthwave _ Chillwave ] _ free-stock-music.com.wav", sr=sr)


# In[ ]:


#Original audio
print("KPOP")
display(Audio(kpop_in1, rate=sr))
display(Audio(kpop_in2, rate=sr))
print("LOFI")
display(Audio(lofi_in1, rate=sr))
display(Audio(lofi_in2, rate=sr))


# In[ ]:


#display audio signal waveforms
plt.figure(figsize=(20,3))
librosa.display.waveplot(kpop_in1, sr=sr)
plt.title("Kpop song 1 waveform", fontsize=20)
plt.xlabel("Time")

plt.figure(figsize=(20,3))
librosa.display.waveplot(kpop_in2, sr=sr)
plt.title("Kpop song 2 waveform", fontsize=20)
plt.xlabel("Time")

plt.figure(figsize=(20,3))
librosa.display.waveplot(lofi_in1, sr=sr)
plt.title("Lofi song 1 waveform", fontsize=20)
plt.xlabel("Time")

plt.figure(figsize=(20,3))
librosa.display.waveplot(lofi_in2, sr=sr)
plt.title("Lofi song 2 waveform", fontsize=20)
plt.xlabel("Time")

# fig.tight_layout() #space the subplots out


# In[ ]:


#plot magnitude spectrums

#kpop song 1
X_mag = 2 * np.abs(np.fft.rfft(kpop_in1)) / len(kpop_in1)
plt.figure(figsize=(20,4))
freq = np.linspace(0, sr/2, int(len(kpop_in1) / 2 + 1)) # Frequency points for plotting
plt.title("Kpop Song 1 Magnitude Spectrum", fontsize=20)
plt.ylabel("Magnitude")
plt.xlabel("Frequency (Hz)")
plt.plot(freq, X_mag)
plt.xlim(0, 1000) #show frequencies from 0 to 1000
plt.ylim(0.0, 0.07) #show magnitude from 0.o to 0.07

#kpop song 2
X_mag = 2 * np.abs(np.fft.rfft(kpop_in2)) / len(kpop_in2)
plt.figure(figsize=(20,4))
freq = np.linspace(0, sr/2, int(len(kpop_in2) / 2 + 1)) # Frequency points for plotting
plt.title("Kpop Song 2 Magnitude Spectrum", fontsize=20)
plt.ylabel("Magnitude")
plt.xlabel("Frequency (Hz)")
plt.plot(freq, X_mag)
plt.xlim(0, 1000) #show frequencies from 0 to 1000
plt.ylim(0.0, 0.07) #show magnitude from 0.o to 0.07

#lofi song 1
X_mag = 2 * np.abs(np.fft.rfft(lofi_in1)) / len(lofi_in1)
plt.figure(figsize=(20,4))
freq = np.linspace(0, sr/2, int(len(lofi_in1) / 2 + 1)) 
plt.title("Lofi Song 1 Magnitude Spectrum", fontsize=20)
plt.ylabel("Magnitude")
plt.xlabel("Frequency (Hz)")
plt.plot(freq, X_mag)
plt.xlim(0, 1000) #show frequencies from 0 to 1000
plt.ylim(0.0, 0.07) #show magnitude from 0.o to 0.07

#lofi song 2
X_mag = 2 * np.abs(np.fft.rfft(lofi_in2)) / len(lofi_in2)
plt.figure(figsize=(20,4))
freq = np.linspace(0, sr/2, int(len(lofi_in2) / 2 + 1)) 
plt.title("Lofi Song 2 Magnitude Spectrum", fontsize=20)
plt.ylabel("Magnitude")
plt.xlabel("Frequency (Hz)")
plt.plot(freq, X_mag )
plt.xlim(0, 1000) #show frequencies from 0 to 1000
plt.ylim(0.0, 0.07) #show magnitude from 0.o to 0.07


# In[ ]:


#extract features from audio
#sr = 44100

kpop1_features = feature_extraction(kpop_in1, srate=sr)
kpop2_features = feature_extraction(kpop_in2, srate=sr)
lofi1_features = feature_extraction(lofi_in1, srate=sr)
lofi2_features = feature_extraction(lofi_in2, srate=sr)

print(kpop1_features)
print(kpop2_features)
print(lofi1_features)
print(lofi2_features)


# | Piece | Zero Cross Mean | Zero cross Std | Centroid Mean | Centroid Std | Bandwidth Mean | Bandwidth Std | Flatness Mean | Flatness Std | Rolloff Mean | Rolloff Std |
# | --- | --- |--- | --- | --- | --- | --- | --- | --- | --- | --- |
# | kpop1 | 0.075 | 0.263 | 3262 | 1040 | 3363 | 582 | 3.87e-3 | 3.72e-2 | 6847 | 2021 |
# | kpop2 | 0.067 | 0.250 | 3364 | 1316 | 3623 | 771 | 1.50e-3 | 2.04e-2 | 7494 | 2668 |
# | lofi1 | 0.037 | 0.188 | 1773 | 2379 | 2618 | 1070 | 2.04e-3 | 7.35e-3 | 3013 | 3652 |
# | lofi2 | 0.028 | 0.164 | 1310 | 884 | 1882 | 727 | 5.65e-4 | 2.27e-2 | 2527 | 2134 |

# In[ ]:


#Define a helper function to normalize the spectral centroid for visualization:
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)


# ### We will now use 1 audio file (lofi song 1) to visualize spectral centroid, bandwidth, flatness and rolloff.

# In[ ]:


#central spectroid
spectral_centroids = librosa.feature.spectral_centroid(lofi_in1 +0.01, sr=sr)[0]
spectral_centroids.shape

frames = range(len(spectral_centroids)) #Compute the time variable for visualization
t = librosa.frames_to_time(frames)

plt.figure(figsize=(20,6))
plt.title("Lofi Song 1 Spectral Centroid (red) and Waveform", fontsize=20)
librosa.display.waveplot(lofi_in1, sr=sr, alpha=0.4) #Plot the spectral centroid along with the waveform:
plt.plot(t, normalize(spectral_centroids), color='r') # normalize for visualization purposes

plt.figure(figsize=(20,6))
plt.title("Lofi Song 1 Spectral Centroid (red) and Waveform (Time = 0:40 to 1:00)", fontsize=20)
librosa.display.waveplot(lofi_in1, sr=sr, alpha=0.4) #Plot the spectral centroid along with the waveform:
plt.plot(t, normalize(spectral_centroids), color='r') # normalize for visualization purposes
plt.xlim(40, 60) #show time from 0:40 to 1:00


# In[ ]:


#zero crossing rate
# https://musicinformationretrieval.com/zcr.html
# zcr = librosa.feature.zero_crossing_rate(lofi_in1)
# plt.figure(figsize=(20,6))
# plt.title("Lofi Song 1 zcr", fontsize=20)
# # librosa.display.waveplot(lofi_in1, sr=sr, alpha=0.4) #Plot the spectral centroid along with the waveform:
# plt.plot(zcr[0]) # normalize for visualization purposes


# In[ ]:


#spectral bandwidth
spectral_bandwidth_2 = librosa.feature.spectral_bandwidth(lofi_in1 +0.01, sr=sr)[0]
spectral_bandwidth_3 = librosa.feature.spectral_bandwidth(lofi_in1 +0.01, sr=sr, p=3)[0]
spectral_bandwidth_4 = librosa.feature.spectral_bandwidth(lofi_in1 +0.01, sr=sr, p=4)[0]

plt.figure(figsize=(20,6))
plt.title("Lofi Song 1 Bandwidth and Waveform", fontsize=20)
librosa.display.waveplot(lofi_in1, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_bandwidth_2), color='r')
plt.plot(t, normalize(spectral_bandwidth_3), color='g')
plt.plot(t, normalize(spectral_bandwidth_4), color='y')
plt.legend(('p = 2', 'p = 3', 'p = 4'))

plt.figure(figsize=(20,6)) #zoomed up version
plt.title("Lofi Song 1 Bandwidth and Waveform (Time = 0:40 to 0:50)", fontsize=20)
librosa.display.waveplot(lofi_in1, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_bandwidth_2), color='r')
plt.plot(t, normalize(spectral_bandwidth_3), color='g')
plt.plot(t, normalize(spectral_bandwidth_4), color='y')
plt.legend(('p = 2', 'p = 3', 'p = 4'))
plt.xlim(40, 50) #show frequencies from 0 to 1000


# In[ ]:


#flatness
spectral_flatness = librosa.feature.spectral_flatness(y=lofi_in1+0.01)[0]

plt.figure(figsize=(20,6))
plt.title("Lofi Song 1 Flatness (red) and waveform", fontsize=20)
librosa.display.waveplot(lofi_in1, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_flatness), color='r')

plt.figure(figsize=(20,6))
plt.title("Lofi Song 1 Flatness (red) and waveform (Time = 0:40 to 1:00)", fontsize=20)
librosa.display.waveplot(lofi_in1, sr=sr, alpha=0.4) #Plot the spectral centroid along with the waveform:
plt.plot(t, normalize(spectral_flatness), color='r') # normalize for visualization purposes
plt.xlim(40, 60) #show time from 0:40 to 1:00


# In[ ]:


#spectral rolloff
spectral_rolloff = librosa.feature.spectral_rolloff(lofi_in1+0.01, sr=sr)[0]

plt.figure(figsize=(20,6))
plt.title("Lofi Song 1 Rolloff (red) and waveform", fontsize=20)
librosa.display.waveplot(lofi_in1, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_rolloff), color='r')

plt.figure(figsize=(20,6))
plt.title("Lofi Song 1 Rolloff (red) and waveform (Time = 0:40 to 1:00)", fontsize=20)
librosa.display.waveplot(lofi_in1, sr=sr, alpha=0.4) #Plot the spectral centroid along with the waveform:
plt.plot(t, normalize(spectral_rolloff), color='r') # normalize for visualization purposes
plt.xlim(40, 60) #show time from 0:40 to 1:00


# In[ ]:


plt.figure(figsize=(20,6))
plt.title("Lofi Song 1", fontsize=20)
librosa.display.waveplot(lofi_in1, sr=sr, alpha=0.4) #Plot the waveform:

plt.plot(t, normalize(spectral_centroids), color='r') # central spectroid
plt.plot(t, normalize(spectral_flatness), color='orange') # flatness
plt.plot(t, normalize(spectral_rolloff), color='y') #

plt.legend(('spectral centroid', 'flatness', 'rolloff'))

plt.xlim(40, 55) #show time from 0:40 to 1:00
plt.ylim(-0.25, 1.00)

