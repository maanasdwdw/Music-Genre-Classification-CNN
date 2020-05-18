"""
Code for preprocessing of tracks.

Commented out part is for processing two tracks at the same time - Used for testing adding only certain part of the audio (ex. vocals) along with either another part or the full audio track.

Line 46 converts stereo to mono to stereo for use with librosa for feature extraction.

Length calculation is done to include as much of data as possible as opposed to trimming the end of the tracks as some are slightly longer than others.

Preprocessed npy arrays can be found in the data folder of the repo.
"""




import numpy as np
from scipy.io import wavfile
import pandas as pd
from sklearn.utils.multiclass import unique_labels
import librosa
from tqdm import tqdm

#path1 = '/music/splits/'
path = '/music/'
hop_length = 512

df = pd.read_csv('/music/genres.csv')
k = unique_labels(df['label'])
k = k.tolist()

def load_mfcc(y,sr,hl=hop_length):
    return librosa.feature.mfcc(y=y, sr=sr, hop_length=hl, n_mfcc=13)
    
def load_spect(y,sr,hl=hop_length):
    return librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hl)
    
def load_chroma_stft(y,sr,hl=hop_length):
    return librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hl)
    
def load_spectral_contrast(y,sr,hl=hop_length):
    return librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hl)
    
def load_as_segments(path):
    for i in tqdm(range(n_samples)):
        rand_class = np.random.choice(class_dist.index,p=prob_dist)
        file = np.random.choice(nf[nf.label==rand_class].index)
        #sr, y = wavfile.read(path1 + f[:-3] + '/other_silent.wav')
        sr, y = wavfile.read(path + file)
        label = nf.at[file,'label']
        rand_index = np.random.randint(0,y.shape[0]-4410)
        sample = y[rand_index:rand_index+4410]

        if len(sample.shape) > 1:
            sample = sample[:, 0]/2.0 + sample[:, 1]/2.0
            sample = sample.astype(float)

        mfcc = load_mfcc(sample,sr)
        spectral_center = load_spect(sample,sr)
        chroma_stft = load_chroma_stft(sample,sr)
        spectral_contrast = load_spectral_contrast(sample,sr)

        mfccs_all_sam_voices.append(mfcc)
        spectral_center_all_sam_voices.append(spectral_center)
        chroma_stft_all_sam_voices.append(chroma_stft)
        spectral_contrast_all_sam_voices.append(spectral_contrast)

        label_sam_voices.append(classes.index(label))
#def load_as_segments(path,path1):
#    for i in tqdm(range(n_samples)):
#        rand_class = np.random.choice(class_dist.index,p=prob_dist)
#        file = np.random.choice(nf[nf.label==rand_class].index)
#        y, sr = librosa.load(path + file, duration=28)
#        sr1, y1 = wavfile.read(path1 + file[:-3] + '/vocals.wav')
#        label = nf.at[file,'label']
#        rand_index = np.random.randint(0,y.shape[0]-2250)
#        sample = y[rand_index:rand_index+2250]
#        sample1 = y1[rand_index:rand_index+2250]
#
#        sample1 = sample1[:, 1].astype(float)
#
#        mfcc = load_mfcc(sample,sr)
#        spectral_center = load_spect(sample,sr)
#        chroma_stft = load_chroma_stft(sample,sr)
#        spectral_contrast = load_spectral_contrast(sample,sr)
#
#        mfcc_voc = load_mfcc(sample1,sr1)
#        spectral_center_voc = load_spect(sample1,sr1)
#        chroma_stft_voc = load_chroma_stft(sample1,sr1)
#        spectral_contrast_voc = load_spectral_contrast(sample1,sr1)
#
#        mfccs_all_sam.append(mfcc)
#        spectral_center_all_sam.append(spectral_center)
#        chroma_stft_all_sam.append(chroma_stft)
#        spectral_contrast_all_sam.append(spectral_contrast)
#
#        mfccs_all_sam_voc.append(mfcc_voc)
#        spectral_center_all_sam_voc.append(spectral_center_voc)
#        chroma_stft_all_sam_voc.append(chroma_stft_voc)
#        spectral_contrast_all_sam_voc.append(spectral_contrast_voc)
#
#        label_sam.append(classes.index(label))


nf = df.set_index('fname', inplace=False)

print("Calculating lengths,")
for f in tqdm(nf.index):
    sr, y = wavfile.read(path + f)
#    sr, y = wavfile.read(path + f[:-3] + '/vocals.wav')
    nf.at[f, 'length'] = y.shape[0]/sr

classes = list(np.unique(nf.label))
class_dist = nf.groupby(['label'])['length'].mean()
n_samples = 2* int(nf['length'].sum()/0.1)
prob_dist = class_dist/class_dist.sum()

label_sam=[]

mfccs_all_sam = []
spectral_center_all_sam = []
chroma_stft_all_sam = []
spectral_contrast_all_sam  = []

#mfccs_all_sam_voc = []
#spectral_center_all_sam_voc = []
#chroma_stft_all_sam_voc = []
#spectral_contrast_all_sam_voc  = []

print("Loading features,")
#load_as_segments(path,path1)
load_as_segments(path)

print("Saving features")
np.save('/music/mfccs_all_sam',mfccs_all_sam)
np.save('/music/spectral_center_all_sam',spectral_center_all_sam)
np.save('/music/spectral_contrast_all_sam',spectral_contrast_all_sam)
np.save('/music/chroma_stft_all_sam',chroma_stft_all_sam)

#np.save('/music/mfccs_all_sam_voc',mfccs_all_sam_voc)
#np.save('music/spectral_center_all_sam_voc',spectral_center_all_sam_voc)
#np.save('/music/spectral_contrast_all_sam_voc',spectral_contrast_all_sam_voc)
#np.save('/music/chroma_stft_all_sam_voc',chroma_stft_all_sam_voc)


np.save('/music/label_sam',label_sam)


