# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 02:28:44 2022

@author: tokay
"""

#%% Açıklama
"""
Amaç : 
    Birden fazla makine öğrenmesi modelini bir arada kullanarak,
    fotoğrataki objeleri tanıyan ve anlamlandıran bir yazılımdır.
    
    Önce el yazısı ile yazılan rakamların fotoğraflarını sisteme yükleyip her bir 
    rakam için sistemimizi eğiticez. Daha sonra sistemimize el yazısı ile yazılan 
    yeni bir rakamı tanımasını isticez.
______________________________________________________________________________________

2 tane machine learnin modeli kullanıldı, bunlar;
    * PCA -Prensible Com. Analysis
    * LR - Lojistic Regresion
______________________________________________________________________________________

Yaklaşık olarak 70 bin adet rakamımız var bu veri setinde 
    * 60 bin tanesi eğitim
    * 10 bin tane de test 
    verisi olarak kullanılacak.
______________________________________________________________________________________

Veriler 28x28 piksel boyutunda kare fotoğraflardır.
______________________________________________________________________________________

1- veri setinin boyutunu düşürücez
2- bu veri seti ile makine öğrenim modelimizi eğiticez
3- eğitim sonrasında test verisi ile test yapıcaz
______________________________________________________________________________________

Veri setinin adresi;
    https://github.com/teavanist/MNIST-JPG

"""

#%% Kütüphaneleri import etme
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 
from sklearn.datasets import fetch_openml #mnist verisetini yüklemek için gerekli
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

mnist = fetch_openml('mnist_784') #datasetini sklearn kütüphanesinden çekiyoruz
# mnist.data.shape => (70000,784) - 70bin tane satır kayıt 768 tanede sütünumuz vardır.

#%% bu veri setinin içerisindeki rakam fotoğraflarını görmek için fonksiyon tanımlıyoruz

#parametre olarak dataframe ve ilgili  fotoğrafının index numarasını alsın.
def showimage(dframe, index):
    some_digit = dframe.to_numpy()[index] #numpy array'e çeviriyoruz
    some_digit_image = some_digit.reshape(28,28) #yeniden boyutlanıdırıyoruz
    
    plt.imshow(some_digit_image,cmap="binary")
    plt.axis("off")
    plt.show()
    
#örnek kullanımı
showimage(mnist.data, 0)

#%% Split Data --> Training Set  ve Test Set

#test ve train oranı - 1/7 ve ve 6/7
# random olarak test ve train olarak ayıracak
train_img, test_img, train_lbl, test_lbl = train_test_split(mnist.data, mnist.target, test_size=1/7.0, random_state=0)
#type(train_img)

#Rakam tahminlerimizi check etmek için train_img dataframe'ini kopyalıyoruz, çünkü biraz sonra değişecektir.
test_img_copy = test_img.copy() #amaç verileri koppyalanmış veriler ile kıyaslamak

showimage(test_img_copy, 0)

#%% Verilerimiz Scale ediyoruz
"""
Çünkü PCA Scale edilmemiş verilerde hatalı sonuçlar verebiliyor bu nedenle mutlaka bu adımı gerçekleştiriyoruz.
Bu amaçla  da StandardScaler kullanıyoruz.
"""
scaler = StandardScaler()

# Scaler'ı sadece training set üzerinde fit yapmamız yeterli
scaler.fit(train_img)

#Ama transform işlemini hem training sete hem de test sete yapmamız gereklidir!
train_img = scaler.transform(train_img)
test_img = scaler.transform(test_img)

#%% PCA işlemini uyguluyoruz
#Variance'nın 95% oranında korunmasını istediğimizi belirtiyoruz

# modelin bir örneğini yapıyoruz
pca = PCA(.95) #bir objeye atıyoruz

#PCA'yı sadece training sete yapmamız yeterlidir
pca.fit(train_img) #

#sonucu gösterelim
print(pca.n_components_)

#Şimdi transform işlemiyle hem train hem de test verisi setimizin  boyutlarını 784'ten 327'ye düşürelim
train_img = pca.transform(train_img)
test_img  = pca.transform(test_img)

#%% 2.Aşama - Logistic Regression 

#default solver çok yavaş çalıştığı için daha hızlı olan 'lfbgs' solverı seçerek logistic regression nesnemizi oluşuturuyoruz
logisticReg = LogisticRegression(solver='lbfgs',max_iter=10000) #max iteri yazmazsak eğer, iteration yetmeyebiliyor ve hata veriyor program

# Logistic Regression modelimizi train datamızı kullanarak eğitiyoruz
logisticReg.fit(train_img, train_lbl) # belirtilen img'nin lbl'ı bu diye eğiticek bu şekilde 60bin tane veriyi kullanarak modelimizi eğitiyoruz

#modelimiz eğitildi şimdi el yazısı rakamları makine öğrenmesi ile tanıma işlemini gerçekleştirelim
logisticReg.predict(test_img[0].reshape(1,-1))#burada bakalım el yazısını modelimiz tanıyor mu? array(['0']) olmalı
showimage(test_img_copy, 0)

logisticReg.predict(test_img[1].reshape(1,-1))
showimage(test_img_copy, 1)

#%% 






















