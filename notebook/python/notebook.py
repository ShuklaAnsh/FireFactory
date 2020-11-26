#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import librosa
import matplotlib.pyplot as plt
import IPython.display as ipd
import mir_eval
import scipy
import sklearn

from pathlib import Path
from random import choices, randint, choice, shuffle, random
from numpy import dot
from numpy.linalg import norm
from scipy.interpolate import interp1d
from heapq import nsmallest


# ## Loading in the data
# > The audio should be placed in the root directory inside a folder named "data". Inside the data folder there should be two folders, "lofi" and "non-lofi". Place the data you want in those folders accordingly

# In[ ]:


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

def load_data(srate):
    '''
    Loads the data from the data folder into numpy data arrays representing the audio files.
    
    params:
        srate: The sample rate to load the data in. Default is 22050 Hz (Good enough for now)
    
    returns: an array of genre arrays. Each genre array contains audio arrays for each audio files 
        [0]: lofi data arrays
        [1]: non-lofi data arrays
    '''
    lofi_paths, non_lofi_paths = _get_paths()
    lofi_data_array = []
    non_lofi_data_array = []
    
    # load in lofi data
    for path in lofi_paths:
        data, _ = librosa.load(path, sr=srate)
        lofi_data_array.append(data)
        
    # load in non-lofi data
    for path in non_lofi_paths:
        data, _ = librosa.load(path, sr=srate)
        non_lofi_data_array.append(data)
    return lofi_data_array, non_lofi_data_array


# In[ ]:


srate = 22050
lofi, non_lofi = load_data(srate)


# In[5]:


ipd.Audio(data=lofi[0], rate=srate)


# ## Pre-processing

# In[5]:


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
    zcr = librosa.feature.zero_crossing_rate(data)
    
    features = [zcr.mean(), zcr.std()]
    
    #SPECTRAL FEATURES
    spectral_features = extract_spectral(data, srate, hop_size)  #store spectral features in array
    
    return np.concatenate([features, spectral_features])


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
    centroid = librosa.feature.spectral_centroid(data, sr=srate, hop_length=hop_length) 
    bandwidth = librosa.feature.spectral_bandwidth(data, sr=srate, hop_length=hop_length)
    flatness = librosa.feature.spectral_flatness(data, hop_length=hop_length)
    rolloff = librosa.feature.spectral_rolloff(data, sr=srate, hop_length=hop_length)
    
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


# In[14]:


#maps a signal of pulse locations 
def get_PLP(data):
    PLP = librosa.beat.plp(data)
    return PLP

#returns tempo and beat event locations in the track
def get_beat_track(data):
    tempo, beats = librosa.beat.beat_track(data)
    return tempo, beats


# In[ ]:

# interpreted from https://www.royvanrijn.com/blog/2010/06/creating-shazam-in-java/
# finds highest magnitude of freq in a most important frequency range 
# produces 5 frequencies in a frame, the frames "fingerprint"
# wholes song has multiple fingerprints and all can be used towards 
ranges = (40, 80, 120, 180, 300)

def getIndex(freq):
    i = 0
    while(ranges[i] < freq):
        i += 1
    return i

def fingerprint_hash(result):
    freqList = []
    for t in range(0, len(result)):
        highScores = {}
        freqNumbers = {}
        for freq in range(0, 300): 
            mag = result[t][freq] 
            index = getIndex(freq)
            if index in highScores:
                if mag > highScores[index]:
                    highScores[index] = mag
                    freqNumbers[index] = freq
            else:
                highScores[index] = mag
                freqNumbers[index] = freq
        
        fingerprint = ''
        for num in freqNumbers:
            fingerprint += str(freqNumbers[num])
        if fingerprint not in freqList:
            freqList.append(fingerprint)
    
    return freqList


# In[13]:


def audio_fingerprint(data, srate):
    '''
    Create the audio fingerprint

    params:
        data: the audio data array of a single audio clip to create the fingerprint of
        
    returns:
        an audio fingerprint of the given data array
    '''
    #corresponds to 4 seconds of audio
    frameSize = 4 * srate
    i = 0

    results = []
    while(i < len(data)):
        frame = data[i:i+frameSize]
        mag = np.abs(np.fft.fft(frame))
        mag = mag[0:int(len(mag)/2)]
        results.append(mag)
        i += frameSize
       
    fingerprint = fingerprint_hash(results)
    return fingerprint


# In[8]:


finger_print = audio_fingerprint(lofi[0], srate)
print(finger_print)


# In[16]:


def create_fingerprint_hashmap(data, paths):
    fingerprint_hashes = {}
    
    for i in range(len(data)):
        song_name = str(paths[i])[13:]
        fingerprint = audio_fingerprint(data[i], srate)
        for a_hash in fingerprint:
            if a_hash in fingerprint_hashes:
                fingerprint_hashes[a_hash].append(song_name)
            else:
                insert = [song_name]
                fingerprint_hashes[a_hash] = insert
                
    return fingerprint_hashes
                


# In[19]:


lofi_paths, non_lofi_paths = _get_paths()
hashmap = create_fingerprint_hashmap(lofi, lofi_paths)
max_len = 0

test_fp = audio_fingerprint(lofi[51], srate)
print(str(lofi_paths[51])[13:])

song_scores = {}

for fingerprint in test_fp:
    song_names = hashmap[fingerprint]
    for song_name in song_names:
        if song_name in song_scores:
            song_scores[song_name] += 1
        else:
            song_scores[song_name] = 1
            
print(song_scores)


# In[ ]:


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

# In[ ]:


#lofi_af = [audio_fingerprint(data) for data in lofi]
non_lofi_af = [audio_fingerprint(data) for data in non_lofi]


# In[ ]:


lofi_mfccs = [generate_mfcc(data) for data in lofi]
non_lofi_mfccs = [generate_mfcc(data) for data in non_lofi]


# ## Sound Bank Generation

# In[ ]:


def generate_soundbank(dataset, srate):
    '''
    Groups similar sounds using K-means clustering
    
    params:
        dataset: The dataset to group sounds for
        srate: sample rate of the data in the dataset
        
    returns:
        groups of the sounds (bass sounds, drum sounds, etc)
    '''
    # pre-onset buffer are the samples before the onset starts
    # post-onset buffer are the samples after the onset starts
    # pre-onset + post-onset = frame size
    pre_onset_buffer = 1025
    post_onset_buffer = 3072
    gaps = []
    segments = []
    features = []
    for data in dataset:
        onset_frames = librosa.onset.onset_detect(data)
        onset_samples = librosa.frames_to_samples(onset_frames)
        for idx in range(len(onset_samples)):
            onset = onset_samples[idx]
            
            # Extract Main Audio Segments
            frame_start = onset - pre_onset_buffer
            frame_end = onset + post_onset_buffer
            if frame_start < 0 or frame_end > len(data):
                continue
            segment = data[frame_start:frame_end]
            segments.append(segment)
            features.append(feature_extraction(segment, srate))
            
            # Extract Background Segments (Gaps)
            if idx+1 >= len(onset_samples):
                continue
            onset = onset_samples[idx]
            start_gap_sample = onset + post_onset_buffer
            end_gap_sample = onset_samples[idx+1] - pre_onset_buffer
            gap = []
            for gap_sample in range(start_gap_sample, end_gap_sample+1):
                gap.append(data[gap_sample])
            gaps.append(gap)
            
    min_max_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1,1))
    scaled_features = min_max_scaler.fit_transform(features)
    clusterer = sklearn.cluster.KMeans(3)
    labels = clusterer.fit_predict(scaled_features)
    
    plt.figure(figsize=(20,5))
    plt.scatter(scaled_features[labels==0,0], scaled_features[labels==0,2], c='r')
    plt.scatter(scaled_features[labels==1,0], scaled_features[labels==1,2], c='g')
    plt.scatter(scaled_features[labels==2,0], scaled_features[labels==2,2], c='b')
    plt.xlabel('Zero Cross Rate (Scaled)')
    plt.ylabel('Spectral Centroid (Scaled)')
    plt.legend(['Class 0', 'Class 1', 'Class 2'])
    plt.show()
    
    sound_groups = [[], [], [], gaps]
    for idx, segment in enumerate(segments):
        if labels[idx] == 0:
            sound_groups[0].append(segment)
        elif labels[idx] == 1:
            sound_groups[1].append(segment)
        else:
            sound_groups[2].append(segment)
    return sound_groups


# In[ ]:


sound_bank = generate_soundbank(lofi, srate)


# In[ ]:


print(len(sound_bank[0]))
ipd.Audio(np.concatenate(sound_bank[0]), rate=srate)


# In[ ]:


print(len(sound_bank[1]))
ipd.Audio(np.concatenate(sound_bank[1]), rate=srate)


# In[ ]:


print(len(sound_bank[2]))
ipd.Audio(np.concatenate(sound_bank[2]), rate=srate)


# In[ ]:


print(len(sound_bank[3]))
ipd.Audio(np.concatenate(sound_bank[3]), rate=srate)


# ## Genetic Algorithm

# In[ ]:


def cos_similarity(genome_mfcc, lofi_mfcc):
    '''
    Computes the cos similarity between a genome and the lofi genre
    
    params:
        genome_mfcc: a generated audio data array
        lofi_mfcc: the MFCC vectors of Lofi and non-Lofi
        
    returns:
        similarity: A similarity rating between the two mfccs
    '''
    similarity = dot(genome_mfcc, lofi_mfcc)/(norm(genome_mfcc)*norm(lofi_mfcc))
    return similarity

def fitness_fn(genome, lofi_mfcc, afs):
    '''
    Determines if the given genome is fit enough
    
    creates a AF and MFCC vector of the genome and compares with the mfccs and afs
    
    params:
        genome: a generated audio data array
        mfccs: the MFCC vectors of Lofi and non-Lofi
        afs: the Audio Fingerprints of Lofi and non-Lofi
        
    returns:
        fitness_value: A rating based on all produced heuristics
    '''
    # Heuristics
    #af = audio_fingerprint(genome)
    genome_samples = np.array([])
    for sound in genome:
        genome_samples = np.append(genome_samples, sound)
    mfcc = generate_mfcc(genome_samples)
    
    fitness_value = 0
    for lofi_mfcc in lofi_mfccs:
        for i in range(len(mfcc)):
            fitness_value += cos_similarity(mfcc[i], lofi_mfcc[i][:len(mfcc[i])])

    return fitness_value/(len(mfcc)*len(lofi_mfccs))

def get_positions(n, max_position):
    '''
    gets n random positions within range max_position
    
    params:
        n: the number of random positions to return
        max_position: the max position in the range
        
    returns:
        positions: alist of positions
    '''
    positions = []
    for i in range(int(n)):
        positions.append(randint(0, max_position))

    return positions


def generate_genome(sound_bank):
    '''
    Creates a random genome from available sounds
    
    params:
        sounds: the possible sounds to be uses for creating genomes
        
    returns:
        genome: a random genome based of available sounds
    '''
    genome_max_length = 50
    genome = []
    total = len(sound_bank[0]) + len(sound_bank[1]) + len(sound_bank[2]) + len(sound_bank[3])
    for sound in sound_bank:
        positions = get_positions(genome_max_length*(len(sound)/total), (len(sound)-1))
        for position in positions:
            if len(genome) < genome_max_length:
                genome.append(sound[position])
                
    
    shuffle(genome)
    return genome


def generate_population(pop_size, sound_bank):
    '''
    Creates pop_size number of genomes from available sounds
    
    params:
        pop_size: the number of genomes to produce
        sounds: the possible sounds to be uses for creating genomes
        
    returns:
        population: a list of genomes with length of pop_size
    '''
    population = []
    
    for i in range(pop_size):
        population.append(generate_genome(sound_bank))
        
    return population


def choose_parents(population, weights):
    '''
    Choose 2 parents from population, better genomes are more likely to be choosen
    
    params:
        population: list of genomes
        weights: list of weights
        
    returns:
        parents: A list of 2 genomes
    '''
    return choices(population, weights=weights, k=2)


def single_point_crossover(parents):
    '''
    Creates 2 new children from sections of both parents
    
    params:
        parents: A list of 2 parents genomes
        
    returns:
        children: 2 new genomes based of a combination of both parents
    '''
    parent_a = parents[0]
    parent_b = parents[1]
    # Ensure a and b have same length
    if len(parent_a) != len(parent_b):
        raise ValueError("Genomes not equal length\n")
        
    split_point = randint(1, len(parent_a)-1)
    child_a = parent_a[:split_point] + parent_b[split_point:]
    child_b = parent_b[:split_point] + parent_a[split_point:]
    children = [child_a, child_b]
    return children


def mutate(genome, sound_bank):
    '''
    Mutates values from a genome at random
    
    params:
        genome: a generated audio data array
        
    returns:
        mutated_genome: a mutated verion of the inputed genome
    '''
    if random() > 0.75:
        rand = randint(0, 109)
        for i in range(10):
            genome_position = randint(0, (len(genome)-1))
            bank_num = randint(0, len(sound_bank)-1)
            sound_position = randint(0, (len(sound_bank[bank_num])-1))
            genome[genome_position] = sound_bank[bank_num][sound_position]
            
    while True:        
        if random() > 0.40:
            genome_position1 = randint(0, (len(genome)-1))
            genome_position2 = randint(0, (len(genome)-1))
            temp = genome[genome_position1]
            genome[genome_position1] = genome[genome_position2]
            genome[genome_position2] = temp
        else:
            break

    return genome


def new_gen(population, weights, sound_bank):
    '''
    Creates the next generations population
    
    params:
        population: list of genomes
        weights: list of weights
        
    returns:
        new_population: A list of genomes ordered by rank
    '''
    new_population = []
    for i in range(int(len(population)/2)):
        parents = choose_parents(population, weights)
        children = single_point_crossover(parents)
        new_population += children
    
    return new_population


def genetic_algorithm(sound_bank, mfccs, afs):
    '''
    Generates a lofi audio clip using a genetic algorithm
    
    The fitness function will use the mfccs and afs to see if its close enough to lofi
    
    params:
        sound_bank: the catagorized, grouped, sounds of lofi to use to make the audio clip.
        mfccs: the MFCC vectors of Lofi and non-Lofi
        afs: the Audio Fingerprints of Lofi and non-Lofi
        
    returns:
        a data array of generated lofi
    '''
    # Number of generations until exit
    generations = 10
    # Generate 6 unique melodies
    population = generate_population(10, sound_bank)
    
    for gen in range(generations):
        # Get comparison mfcc
        lofi_mfccs = []
        for i in range(5):
            lofi_mfccs.append(choice(mfccs))
        
        # Get weights
        weights = []
        for genome in population:
            weights.append(fitness_fn(genome, lofi_mfccs, afs))
            
        # Create next generation genomes
        population = new_gen(population, weights, sound_bank)
        weights = []
        for genome in population:
            weights.append(fitness_fn(genome, lofi_mfccs, afs))
        worst = nsmallest(5, weights)
        for i in range(len(weights)):
            if weights[i] in worst:
                mutate(population[i], sound_bank)
        
        
    weights = []
    for genome in population:
        weights.append(fitness_fn(genome, lofi_mfccs, afs))
    return population, weights


# In[ ]:


population, weights = genetic_algorithm(sound_bank, lofi_mfccs, [])


# In[ ]:


print(weights)
genome_samples = np.array([])
for sound in population[6]:
    genome_samples = np.append(genome_samples, sound)
ipd.Audio(genome_samples, rate=22050)


# In[ ]:


import soundfile
soundfile.write('weight_0.21.wav', genome_samples, 22050)


# In[ ]:




