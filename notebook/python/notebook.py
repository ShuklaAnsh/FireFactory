#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import librosa
import matplotlib.pyplot as plt
import IPython.display as ipd
from pathlib import Path


# ## Loading in the data
# > The audio should be placed in the root directory inside a folder named "data". Inside the data folder there should be two folders, "lofi" and "non-lofi". Place the data you want in those folders accordingly

# In[14]:


def _get_paths():
    '''
    Not meant to be called externally. Used by load_data.
    
    Prepares the file paths for the audio files in the data folder (lofi and non-lofi directories)
    for data load
    '''
    lofi_file_paths = []
    non_lofi_file_paths = []
    lofi_basepath = Path("../data/lofi")
    non_lofi_basepath = Path("../data/non-lofi")
    lofi_files_in_basepath = lofi_basepath.iterdir()
    non_lofi_files_in_basepath = non_lofi_basepath.iterdir()
    # Iterate over items in the lofi folder and save filepaths
    for lofi_item in lofi_files_in_basepath:
        if lofi_item.is_file():
            lofi_file_paths.append(lofi_item)
    # Iterate over items in the non-lofi folder and save filepaths
    for non_lofi_item in non_lofi_file_paths:
        if non_lofi_item.is_file():
            non_lofi_file_paths.append(non_lofi_item)
    
    return lofi_file_paths, non_lofi_file_paths


def load_data(srate=22050):
    '''
    Loads the data from the data folder into numpy data arrays representing the audio files.
    
    params:
        sr: The sample rate to load the data in. Default is 22050 Hz (Good enough for now)
    
    returns: an array of genre arrays. Each genre array contains audio arrays for each audio files 
        [0]: lofi data arrays
        [1]: non-lofi data arrays
    '''
    lofi_paths, non_lofi_paths = _get_paths()
    lofi_data_array = []
    non_lofi_data_array = []
    sr = 0
    
    # load in lofi data
    for path in lofi_paths:
        data, sr = librosa.load(path, sr=srate)
        lofi_data_array.append(data)
        
    # load in non-lofi data
    for path in non_lofi_paths:
        data, sr = librosa.load(path, sr=srate)
        non_lofi_data_array.append(data)
    return lofi_data_array, non_lofi_data_array


# In[15]:


srate = 22050
lofi, non_lofi = load_data(srate)


# In[16]:


ipd.Audio(data=lofi[0], rate=srate)


# ## Pre-processing

# In[23]:


def feature_extraction(data):
    '''
    Extracts the features
    
    This will be used to organize the different sounds in lofi music (drums, bass, high_notes, etc)
    
    params:
        data: the audio data array of a single audio clip to extract features from
        
    returns:
        an array of features
    '''
    
    srate = 22050 
    hopSize = 512 
    
    #SPECTRAL FEATURES
    spectral_features = extract_spectral(data,srate,hopSize)  #store spectral features in array
    
    pass


# In[ ]:


# Taken from Jordie's 'Audio Feature Extraction' notebook
# called in 'feature_extraction(data)'
def extract_spectral(data, sr, hop_length):
    '''
    Extracts spectral features
    
    Called in 'feature_extraction(data)'
    
    Taken from Jordie's 'Audio Feature Extraction' notebook

    params:
        data: the audio data array of a single audio clip to extract features from
        sr: sample rate
        hop_length: hop length
        
    returns:
        an list of np arrays called a feature vector
    '''
    # np.array
    centroid = librosa.feature.spectral_centroid(data, sr=sr, hop_length=hop_length) 
    bandwidth = librosa.feature.spectral_bandwidth(data, sr=sr, hop_length=hop_length)
    flatness = librosa.feature.spectral_flatness(data, hop_length=hop_length)
    rolloff = librosa.feature.spectral_rolloff(data, sr=sr, hop_length=hop_length)
    
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


# In[24]:


def audio_fingerprint(data):
    '''
    Create the audio fingerprint
    
    
    params:
        data: the audio data array of a single audio clip to create the fingerprint of
        
    returns:
        an audio fingerprint of the given data array
    '''
    f, t, sxx = scipy.signal.spectrogram(data)
    peaks = scipy.signal.find_peaks(f, height, threshold, distance, prominence)
    fingerprint = fingerprint_hash(peaks)
    pass


# In[25]:


def generate_mfcc(data):
    '''
    Create the mfcc feature vector

    params:
        data: the audio data array of a single audio clip to create the MFCC vector for
        
    returns:
        an MFCC vector of the given vector
    '''
    
    mfcc_data = librosa.feature.mfcc(data)
    return mfcc_data


# ## Data Organization

# In[26]:


lofi_af = [audio_fingerprint(data) for data in lofi]
non_lofi_af = [audio_fingerprint(data) for data in non_lofi]


# In[27]:


lofi_mfccs = [generate_mfcc(data) for data in lofi]
non_lofi_mfccs = [generate_mfcc(data) for data in non_lofi]


# In[28]:


lofi_features = [feature_extraction(data) for data in lofi]

## sounds = do_some_stuff_with_features


# ## Sound Bank Generation

# In[30]:


def k_means_clustering(sounds):
    '''
    Groups similar sounds using K-means clustering
    
    params:
        sounds: the sounds in lofi music (drums, bass, high_notes, etc)
        
    returns:
        groups of the sounds (bass sounds, drum sounds, etc)
    '''
    pass


# ## Genetic Algorithm

# In[31]:


def fitness_fn(genome, mfccs, afs):
    '''
    Determines if the given genome is fit enough
    
    creates a AF and MFCC vector of the genome and compares with the mfccs and afs
    
    params:
        genome: a generated audio data array
        mfcss: the MFCC vectors of Lofi and non-Lofi
        afs: the Audio Fingerprints of Lofi and non-Lofi
        
    returns:
        a data array of generated lofi
    '''
    af = audio_fingerprint(genome)
    mgcc = generate_mfcc(genome)
    pass

def genetic_algorithm(sound_bank, mfccs, afs):
    '''
    Generates a lofi audio clip using a genetic algorithm
    
    The fitness function will use the mfccs and afs to see if its close enough to lofi
    
    params:
        sound_bank: the catagorized, grouped, sounds of lofi to use to make the audio clip.
        mfcss: the MFCC vectors of Lofi and non-Lofi
        afs: the Audio Fingerprints of Lofi and non-Lofi
        
    returns:
        a data array of generated lofi
    '''
    pass


# In[ ]:




