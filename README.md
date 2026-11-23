# Music and Audio Machine Learning Projects

A collection of my machine learning projects on audio signals and music. The projects follow a natural progression: extracting features from sound, classifying music by genre, and generating new music with AI.

| Project | Topic | Tools |
|---|---|---|
| [Audio-Feature-Extraction](Audio-Feature-Extraction) | Feature extraction from audio signals: waveform, STFT, zero-crossing rate, spectral centroid, MFCC | Python, librosa |
| [Music-Classification](Music-Classification) | Classifying 10 music genres using the GTZAN dataset | Python, librosa, scikit-learn, TensorFlow/Keras, Jupyter |
| [Music-Generation-with-LSTM](Music-Generation-with-LSTM) | Generating melodies with an LSTM model trained on Beethoven MIDI files | Python, TensorFlow/Keras, music21, Jupyter |

## Structure

```
Music-and-Audio-MachineLearning-Projects/
├── Audio-Feature-Extraction/     # feature extraction walkthrough with librosa, sample audio files
├── Music-Classification/         # genre classification notebook and code
└── Music-Generation-with-LSTM/   # music generation notebook, MIDI files
```

## Dataset

The Music-Classification project uses the [GTZAN dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification). Because of its size (about 1.2 GB), the dataset is not included in this repository. Setup instructions are in [Music-Classification/README.md](Music-Classification/README.md).

## History

This repository was created by merging three previously separate repositories, with their full commit histories preserved:

- **Audio-Feature-Extraction:** May 4, 2022 – May 18, 2022
- **Music-Classification:** May 4, 2022 – May 19, 2022
- **Music-Generation-with-LSTM:** May 9, 2022 – May 18, 2022
