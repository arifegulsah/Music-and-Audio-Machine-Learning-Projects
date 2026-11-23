# Music-Classification-with-MLP
Classification of music genres using librosa and Keras

[Koda açıklamalar ile beraber erişmek için](MusicGenreClassification.ipynb)
<br>
[Sadece koda erişmek için](wholeCode.py)

Bu çalışmadaki amacım, her birinin 100 farklı örneği olan 10 farklı müzik türündeki ses dosyalarını **Keras** ile oluşturduğum bir modele eğitmek ve bir müziğin hangi türe ait olduğunu tahmin edebilmekti.

[Audio-Feature-Extraction](../Audio-Feature-Extraction) yazımda ses sinyallerinden nasıl öznitelik çıkarılabileceğini anlatmıştım. Bu projede o yöntemleri bir veri setinin tamamına uygulayıp, elde ettiğim değerlerle bir sınıflandırma modeli eğittim.

## VERİ SETİ

Bu projede [GTZAN veri setini](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) kullandım. Veri setinde *blues, classical, country, disco, hiphop, jazz, metal, pop, reggae* ve *rock* olmak üzere 10 müzik türü ve her türden 100 adet 30 saniyelik ses dosyası bulunmaktadır.

Veri seti boyutu nedeniyle repoya eklenmemiştir. Kodu çalıştırmak isterseniz veri setini Kaggle'dan indirip `Data/genres_original` klasörünü bu klasörün içine `genres` adıyla koymanız yeterli:

```
Music-Classification-with-MLP/
└── genres/
    ├── blues/
    ├── classical/
    └── ...
```

## ÖZNİTELİK ÇIKARIMI

Verilerimizi modelde eğitmeden önce ön işlemesini yapmak durumundayız. Bunun için ses verilerinde uygulanabilecek belli başlı yöntemler bulunmaktadır. Bu projede kullandığım yöntemler:

1. Kromagramın hesaplanması (*chroma_stft*),
2. Spektrogramdan her kare için ortalama karekökün hesaplanması (*rmse*),
3. Spektral sentroidin hesaplanması (*spectral_centroid*),
4. Spektral bant genişliğinin hesaplanması (*spectral_bandwidth*),
5. Devre dışı bırakma frekansının hesaplanması (*rolloff*),
6. Sıfır geçiş oranının hesaplanması (*zero_crossing_rate*),
7. 20 adet MFCC değerinin hesaplanması (*mfcc1 – mfcc20*).

Daha detaylı bilgiyi librosa paketinin [dokümantasyonunu](https://librosa.org/doc/latest/index.html) okuyarak edinebilirsiniz.

Çıkardığım değerleri incelemek ve üzerlerinde işlem yapabilmek için manuel şekilde bir csv dosyası oluşturdum. Bu dosyada *filename* sütununda ses dosyalarının isimleri, yukarıdaki yöntemlerin sütunlarında **featurelarımız**, *label* sütununda ise **classlarımız** yer alıyor.

```
header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
for i in range(1, 21):
    header += f' mfcc{i}'
header += ' label'
header = header.split()

csvfilepath = 'csvfile/audioDatas.csv'
```

Şimdi sıra oluşturduğumuz csv dosyasını doldurmakta. Aşağıdaki döngü bütün ses dosyalarını tek tek yükleyip özniteliklerini çıkarıyor ve her yöntemin ortalama değerini dosyaya yazıyor. Bu işlem biraz uzun sürebilir.

```
file = open(csvfilepath, 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)
genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
for g in genres:
    for filename in os.listdir(f'genres/{g}'):
        songname = f'genres/{g}/{filename}'
        y, sr = librosa.load(songname, mono=True, duration=30)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        rmse = librosa.feature.rms(y=y)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
        for e in mfcc:
            to_append += f' {np.mean(e)}'
        to_append += f' {g}'
        file = open(csvfilepath, 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())
```

İşlem bittikten sonra yeni veri setimizi inceleyelim. Oluşan [audioDatas.csv](csvfile/audioDatas.csv) dosyasında 1000 satır ve 28 sütun bulunuyor. İlk 5 satırı şu şekilde:

| filename | chroma_stft | rmse | spectral_centroid | spectral_bandwidth | rolloff | zero_crossing_rate | mfcc1 | mfcc2 | ... | mfcc20 | label |
|---|---|---|---|---|---|---|---|---|---|---|---|
| blues.00000.au | 0.349951 | 0.130192 | 1784.416546 | 2002.657106 | 3806.418650 | 0.083066 | -113.619385 | 121.553017 | ... | 1.219947 | blues |
| blues.00001.au | 0.340945 | 0.095892 | 1529.871314 | 2038.612143 | 3548.986873 | 0.056044 | -207.581512 | 123.997147 | ... | 0.532144 | blues |
| blues.00002.au | 0.363562 | 0.175494 | 1552.637786 | 1747.382028 | 3041.089944 | 0.076301 | -90.776344 | 140.448608 | ... | -2.231391 | blues |
| blues.00003.au | 0.404848 | 0.141139 | 1070.110059 | 1596.244204 | 2185.061787 | 0.033309 | -199.462006 | 150.094727 | ... | -3.406940 | blues |
| blues.00004.au | 0.308598 | 0.091540 | 1835.507009 | 1748.367477 | 3581.003346 | 0.101500 | -160.291855 | 126.195763 | ... | -11.701833 | blues |

Görüldüğü üzere ses dosyalarımız için öznitelikler çıkarılmış ve sayısal değerler olarak kaydedilmiş.

## VERİ ÖN İŞLEME

Modeli eğitirken bu sayısal değerlerden yararlanacağız, lakin dosya isimlerinin tutulduğu *filename* sütunu modeli eğitirken hiçbir işe yaramayacaktır. O sütunu droplayarak devre dışı bırakıyoruz.

```
data = pd.read_csv(csvfilepath)
data = data.drop(['filename'],axis=1)
```

### Label Encoding

Veri ön işleme adımlarından birisi olan "label encoder" olayı kategorik olan her verinin sayısallaştırılmasına denir. Bizim veri setimizdeki kategorik tipteki tek sütun *label* isimli sütunumuz.

```
genre_list = data.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)
print(y)
```

Sonucu yazdırdığımız zaman bütün class isimlerimizin 0 ile 9 arasında numaralara atandığını görüyoruz:

| blues | classical | country | disco | hiphop | jazz | metal | pop | reggae | rock |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |

### Feature Scaling

Veri ön işleme adımlarından sonuncusu olan öznitelik ölçeklendirme (feature scaling) işlemimizi de gerçekleştiriyoruz. Bunu yapmamızın sebebi, modelin büyük aralıklı değişkenlerin küçük aralıklı değişkenlerden daha önemli olduğunu düşünmesini engellemek. Örneğin yukarıdaki tabloda *spectral_centroid* değerleri binlerle ifade edilirken *zero_crossing_rate* değerleri 0 ile 1 arasında.

Ardından X ve y değerlerimizi `train_test_split` ile %80 eğitim ve %20 test olacak şekilde ayırıyoruz. Bu sayede ön işleme aşamamız bitmiş oluyor.

```
scaler = StandardScaler()
X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

## MODELİN OLUŞTURULMASI

Modelimizi oluştururken *keras* kütüphanesinden faydalanacağız. Sequential ile modelimizi oluşturuyoruz, add ile katmanlarını ekliyoruz ve compile ile de parametrelerini veriyoruz.

```
model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

Son katmanda 10 nöron var çünkü 10 farklı müzik türünü sınıflandırıyoruz. `model.summary()` ile modelin yapısına baktığımızda çıktım şu şekilde:

| Katman | Çıktı Boyutu | Parametre Sayısı |
|---|---|---|
| dense (Dense) | (None, 256) | 6.912 |
| dense_1 (Dense) | (None, 128) | 32.896 |
| dense_2 (Dense) | (None, 64) | 8.256 |
| dense_3 (Dense) | (None, 10) | 650 |
| **Toplam** | | **48.714** |

## MODELİN EĞİTİLMESİ

Artık modelimizi eğitebiliriz. Bunun için fit fonksiyonundan faydalanıyoruz.

```
history = model.fit(X_train,
                    y_train,
                    epochs=200,
                    batch_size=256)
```

200 epoch sonunda modelin eğitim verisi üzerindeki doğruluğu **%99,87** oldu.

## MODELİN DEĞERLENDİRİLMESİ

Modelimizi daha önce hiç görmediği test verisi ile değerlendiriyoruz.

```
test_loss, test_acc = model.evaluate(X_test,y_test)
print('test_acc: ',test_acc)
```

Çıktım şu şekilde:

```
test_acc:  0.6800000071525574
```

Tahminlerimizi *predictions* ismini verdiğimiz değişkende tutuyoruz. Bu değişkeni incelediğimiz zaman her bir ses dosyasının, her bir müzik türüne olan yakınlığını görmekteyiz. Burada bizim için önemli olan şey, modelin hesapladığı en yakın türü bulmaktır. Bunu argmax kullanarak yapabiliriz.

```
predictions = model.predict(X_test)

maxpredicts = []
for element in predictions:
    temp = np.argmax(element)
    maxpredicts.append(temp)
    
y_pred = np.array(maxpredicts)    
```

*y_pred* ismini verdiğimiz değişkende artık modelin tahmin ettiği sınıflar bulunmakta. Sonuçlarımızı kolayca değerlendirmek için precision, recall ve f1 score gibi metrikleri döndüren `classification_report` fonksiyonundan yararlanıyoruz. Sonuçları daha düzenli görebilmek için de bir csv dosyasına aktarıyoruz.

```
target_names = ['0blues', '1classical', '2country', '3disco', '4hiphop', '5jazz', '6metal', '7pop', '8reggae', '9rock']

clsf_report = pd.DataFrame(classification_report(y_true = y_test, y_pred = y_pred,target_names=target_names, output_dict=True)).transpose()
clsf_report.to_csv('csvfile/classificationReport.csv', index= True)
```

Çıktım şu şekilde ([ClassificationReport.csv](csvfile/ClassificationReport.csv)):

| Tür | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| blues | 0.68 | 0.71 | 0.70 | 21 |
| classical | 0.86 | 0.82 | 0.84 | 22 |
| country | 0.58 | 0.61 | 0.59 | 18 |
| disco | 0.63 | 0.71 | 0.67 | 24 |
| hiphop | 0.78 | 0.58 | 0.67 | 24 |
| jazz | 0.65 | 0.68 | 0.67 | 19 |
| metal | 0.74 | 0.88 | 0.80 | 16 |
| pop | 0.89 | 0.73 | 0.80 | 22 |
| reggae | 0.61 | 0.65 | 0.63 | 17 |
| rock | 0.39 | 0.41 | 0.40 | 17 |
| **accuracy** | | | **0.68** | **200** |
| macro avg | 0.68 | 0.68 | 0.68 | 200 |
| weighted avg | 0.69 | 0.68 | 0.68 | 200 |

## SONUÇLAR

Modelimiz test verisinde **%68** doğruluk elde etti. 10 sınıflı bir problemde rastgele tahminin doğruluğu %10 olacağı için model müzik türleri arasındaki farkları büyük ölçüde öğrenmiş diyebiliriz.

- En başarılı tahmin edilen türler **classical** (0.84), **metal** ve **pop** (0.80) oldu. Bu türlerin ses karakteristikleri diğerlerinden daha belirgin ayrışıyor.
- En zorlanılan tür ise **rock** (0.40) oldu. Rock müzik; metal, country ve blues gibi türlerle benzer özellikler taşıdığı için model bu türleri birbirine karıştırabiliyor.
- Modelin eğitim verisindeki doğruluğu %99,87 iken test verisindeki doğruluğu %68. Bu fark modelin eğitim verisini ezberlediğini (overfitting) gösteriyor. Dropout katmanları eklemek veya epoch sayısını azaltmak bu farkı kapatmaya yardımcı olabilir.

`train_test_split` fonksiyonunda sabit bir `random_state` verilmediği için kodu her çalıştırdığınızda veriler farklı şekilde bölünür, bu yüzden sonuçlarınız birkaç puan farklı çıkabilir.
