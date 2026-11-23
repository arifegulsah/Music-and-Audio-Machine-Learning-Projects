# Music and Audio Machine Learning Projects

Ses sinyalleri ve müzik üzerine yaptığım makine öğrenmesi çalışmaları. Projeler, sesten öznitelik çıkarmaktan müzik türü sınıflandırmaya ve yapay zekâ ile müzik üretmeye uzanan bir sıra izliyor.

| Proje | Konu | Araçlar |
|---|---|---|
| [Audio-Feature-Extraction](Audio-Feature-Extraction) | Ses sinyallerinden öznitelik çıkarma: dalga formu, STFT, sıfır geçiş oranı, spektral merkez, MFCC | Python, librosa |
| [Music-Classification](Music-Classification) | GTZAN veri seti ile 10 müzik türünün sınıflandırılması | Python, librosa, scikit-learn, TensorFlow/Keras, Jupyter |
| [Music-Generation-with-LSTM](Music-Generation-with-LSTM) | Beethoven eserlerinin MIDI dosyalarıyla eğitilen LSTM modeli ile melodi üretme | Python, TensorFlow/Keras, music21, Jupyter |

## Yapı

```
Music-and-Audio-MachineLearning-Projects/
├── Audio-Feature-Extraction/     # librosa ile öznitelik çıkarma anlatımı ve örnek ses dosyaları
├── Music-Classification/         # müzik türü sınıflandırma notebook'u ve kodu
└── Music-Generation-with-LSTM/   # LSTM ile müzik üretimi notebook'u, MIDI dosyaları
```

## Veri seti

Music-Classification projesi [GTZAN veri setini](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) kullanır. Veri seti boyutu nedeniyle (yaklaşık 1,2 GB) bu repoya eklenmedi; kurulum adımları [Music-Classification/README.md](Music-Classification/README.md) dosyasında.

## Geçmiş

Bu repo, daha önce ayrı ayrı tutulan üç reponun commit geçmişleri korunarak birleştirilmesiyle oluşturuldu.
