# AI-El Yazısını Tanıma
### Fotoğraflardaki El Yazını Yapay Zeka İle Otomatik Tanıma Yazılımı

___

### Amaç : 
Birden fazla makine öğrenmesi modelini bir arada kullanarak, fotoğrataki objeleri tanıyan ve anlamlandıran bir yazılımdır.
    
Önce el yazısı ile yazılan rakamların fotoğraflarını sisteme yükleyip her bir rakam için sistemimizi eğiticez. Daha sonra sistemimize el yazısı ile yazılan yeni bir rakamı tanımasını isticez.

___

2 tane Machine Learning modeli kullanıldı, bunlar;

    - PCA: Principal Component Analysis
    - LR : Logistic Regression
___

Yaklaşık olarak 70 bin adet rakamımız var bu veri setinde 
    
    * 60 bin tanesi eğitim,
    * 10 bin tane de test 

verisi olarak kullanıldı.
    
    * Veriler 28x28 piksel boyutunda kare fotoğraflardır.
___
### Hedefler
    1- Veri setinin boyutunu indirgiyecez.
    2- Bu veri seti ile makine öğrenim modelimizi eğitecez.
    3- Eğitim sonrasında test verisi ile test yapacaz.
    4- Ve son olarak Accuracy değerimizi ölçücez.
___

## Sonuç Ve Değerlendirme
Bu projede PCA kullanarak Logistic Regression tarafından yapay zekanın eğitilme süresini önemli ölçüde kısalttık. Ben 95% vaarriance korumayı hedefliyerek projeyi gerçekleştirdim. Daha düşük variance lerde sürenin belli bir ölçüde kısalması görülebilir.  10 tane digit için yapay zekanın eğitim süresini çok büyük ölçüde kısalttan PCA algoritması yüzlerce hatta binlerce değişik nesne tipi için yapay zekanın eğitim süresini saatler mertebesinde kısaltacak ve bu da programlarımızın çok daha hızlı çalışmasını sağlıyacaktır.

Bu proje ile tamamen birbirinden farklı 2 makine öğrenme modelini bir araya getirip günlük karşımıza çıkabilecek birr işi bilgisayarımıza yapay zeka ile gerçekleştirdik.


___
### Veri setinin adresi;
* [MNIST-JPG](https://github.com/teavanist/MNIST-JPG)

___

## İletişim

* [Özgür Tokay](mailto:ozytky@gmail.com)

___

## Teşekkürler
* [İbrahim Türkoğlu](http://ibrahimturkoglu.com/?page_id=21)