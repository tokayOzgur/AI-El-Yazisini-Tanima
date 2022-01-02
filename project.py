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
train_img, test_img, train_lbl, test_lbl = train_test_split(mnist.data, mnist.target, test_size=1/7.0, random_state=0)
type(train_img)

#Rakam tahminlerimizi check etmek için train_img dataframe'ini kopyalıyoruz, çünkü biraz sonra değişecektir.
test_img_copy = test_img.copy()

showimage(test_img_copy, 0)






























