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
from sklearn.preprocessing import StandarScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split




































