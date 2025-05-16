
![Logo](static/logo.png)

# TIMITPhones: TIMIT Phoneme Dataset
This repository contains a PyTorch ```Dataset``` builder of all sliced phonemes extracted from the TIMIT [[1]](https://catalog.ldc.upenn.edu/LDC93S1) dataset. It includes audio, ```.csv```, and code files.


## Deployment
To load the dataset, simply clone the repository, unzip the ```timit-phones.zip``` file, and use the following:

```
dataset = TIMITPhones(data_dir='timit-phones', sample_len=16000)
```

The dataset object automatically applies all the necessary preprocessing steps such as truncating, padding, and converting to mono. Truncating and padding depend on the ```sample_len``` variable: if the clip is longer, it will get truncated; if shorter, it will be padded so no shape mismatches arise. The default is set to 16,000 samples (1s clips).

Each example returns two tensors: a **waveform** (shape ```[1 x sample_len]```) and its **encoded target**.
```
example, label = dataset[0]
print(f"Audio shape: {example.shape}")
print(f"Label: {label}")

Output:
Audio shape: torch.Size([1, 16000])
Label: 27
```

### Encodings Reference
There are 61 unique phoneme classes in the dataset. The mappings are provided below.
```
'aa': 0, 'ae': 1, 'ah': 2, 'ao': 3, 'aw': 4, 'ax': 5, 'ax-h': 6, 'axr': 7, 
'ay': 8, 'b': 9, 'bcl': 10, 'ch': 11, 'd': 12, 'dcl': 13, 'dh': 14, 'dx': 15, 
'eh': 16, 'el': 17, 'em': 18, 'en': 19, 'eng': 20, 'epi': 21, 'er': 22, 'ey': 23, 
'f': 24, 'g': 25, 'gcl': 26, 'h#': 27, 'hh': 28, 'hv': 29, 'ih': 30, 'ix': 31, 
'iy': 32, 'jh': 33, 'k': 34, 'kcl': 35, 'l': 36, 'm': 37, 'n': 38, 'ng': 39, 
'nx': 40, 'ow': 41, 'oy': 42, 'p': 43, 'pau': 44, 'pcl': 45, 'q': 46, 'r': 47, 
's': 48, 'sh': 49, 't': 50, 'tcl': 51, 'th': 52, 'uh': 53, 'uw': 54, 'ux': 55, 
'v': 56, 'w': 57, 'y': 58, 'z': 59, 'zh': 60
```

In addition, the repository includes the mapping dictionary ```.pkl``` file. The file can be opened using the ```load_mappings()``` functionality located in ```timit.py```.


## Authors

- [@IParraMartin](https://github.com/IParraMartin)


## References
[1] Garofolo, John S., et al. TIMIT Acoustic-Phonetic Continuous Speech Corpus LDC93S1. Web 
Download. Philadelphia: Linguistic Data Consortium, 1993.