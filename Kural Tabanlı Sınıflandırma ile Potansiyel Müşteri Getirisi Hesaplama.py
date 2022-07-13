#Seviye tabanlı (level based) yeni müşteri tanımları (persona) oluşturulacak
#Bu yeni müşteri tanımlarına göre segmentler oluşturulacak
#Bu segmentlere göre yeni gelebilecek müşterilerin şirkete ortalama ne kadar kazandırabileceği tahmin edilecek

# persona.cv = ürünlerin fiyatlarını ve bu ürünleri satın alan kullanıcıların bazı demografik bilgileri yer alır.
# Her satış işleminde oluşan kayıtlardan meydana gelmektedir; tablo tekilleştirilmemiştir.

## Değişkenler :
# PRICE : Müşterinin harcama tutarı
# SOURCE : Müşterinin bağlandığı cihaz türü
# SEX : Müşterinin cinsiyeti
# COUNTRY : Müşterinin ülkesi
# AGE : Müşterinin yaşı


import numpy as np
import pandas as pd
import seaborn as sns
from pandas import DataFrame

persona_veriseti = pd.read_csv("C:/Users/emel.filiz/Desktop/Veri Mühendisleri İçin Python Programlama/DERS NOTLARIM/BİTİRME PROJESİ/persona.csv")

#veri seti temel bilgileri

def check_df(dataframe, head = 5):
    print("########################### Shape ######################################")
    print(dataframe.shape)
    print("########################### Types ######################################")
    print(dataframe.dtypes)
    print("########################### Head ######################################")
    print(dataframe.head(head))
    print("########################### Tail ######################################")
    print(dataframe.tail(head))
    print("########################### NA ######################################")
    print(dataframe.isnull().sum())
    print("########################### Quantiles ######################################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(persona_veriseti)

#kaç tane unique source var?
persona_veriseti["SOURCE"].unique()

#kaç tane unique price var?
persona_veriseti["PRICE"].unique()

#hangi pricedan kaçar tane satış gerçekleşmiş ?
persona_veriseti.groupby("PRICE").count()

#hangi ülkeden kaçar tane satış gerçekleşmiş ?
persona_veriseti.groupby("COUNTRY").count()

#ülkelere göre satışlardan toplam ne kadar kazanılmış?
persona_veriseti.groupby("COUNTRY")["PRICE"].sum()

#SOURCE türlerine göre satış sayıları nedir?
persona_veriseti.groupby("SOURCE")["PRICE"].count()

#Ülkelere göre PRICE ortalamaları nedir?
persona_veriseti.groupby("COUNTRY")["PRICE"].mean()

#SOURCE'lara göre PRICE ortalamaları nedir?
persona_veriseti.groupby("SOURCE")["PRICE"].mean()

#COUNTRY-SOURCE kırılımında PRICE ortalamaları nedir?
persona_veriseti.groupby(["COUNTRY","SOURCE"])["PRICE"].mean()

#COUNTRY, SOURCE, SEX, AGE kırılımında ortalama kazançlar nedir?
persona_veriseti.groupby(["COUNTRY","SOURCE","SEX","AGE"])["PRICE"].mean()

#Çıktıyı azalan olacak şekilde PRICE’a göre sıralayınız.Çıktıyı agg_df olarak kaydediniz.
agg_df = persona_veriseti.groupby(["COUNTRY","SOURCE","SEX","AGE"])["PRICE"].mean().sort_values(ascending=False)
agg_df.head()

#Indekste yer alan isimleri değişken ismine çeviriniz.
#Üçüncü sorunun çıktısında yer alan PRICE dışındaki tüm değişkenler index isimleridir. Bu isimleri değişken isimlerine çeviriniz.

agg_df.index
agg_df: object=agg_df.reset_index()

#Age değişkenini kategorik değişkene çeviriniz ve agg_df’e ekleyiniz.
#Age sayısal değişkenini kategorik değişkene çeviriniz.
#Aralıkları ikna edici şekilde oluşturunuz.
#Örneğin: ‘0_18', ‘19_23', '24_30', '31_40', '41_70'

agg_df["AGE"] --> sonuç : dtype: int64 geldi. yani age değişkeni kategorik değildir, integer tipindedir.

#kategorik değişkenleri yakalama
kategorik_degiskenler = [column for column in agg_df.columns if str(agg_df[column].dtypes) in ["category","object","bool"]]
kategorik_degiskenler --> SONUÇ : ['COUNTRY', 'SOURCE', 'SEX'] --> AGE değişkeni yok. buradan da age değişkeninin kategorik değişken olmadığını gördük.

#o halde age değişkenini kategorik olmasına rağmen integer görünümlü olarak görüyoruz.

agg_df["AGE_CAT"] = agg_df["AGE"].astype("category")
label_isimleri=["0_18","19_23","24_30","31_40","41_70"]
agg_df["AGE_CAT"] = pd.cut(agg_df["AGE_CAT"], [0,18,24,30,40,70], labels=label_isimleri)
agg_df.head(30)

#Yeni seviye tabanlı müşterileri (persona) tanımlayınız ve veri setine değişken olarak ekleyiniz.
#Yeni eklenecek değişkenin adı: customers_level_based
#Önceki soruda elde edeceğiniz çıktıdaki gözlemleri bir araya getirerek customers_level_based değişkenini oluşturmanız gerekmektedir.
#Dikkat! List comprehension ile customers_level_based değerleri oluşturulduktan sonra bu değerlerin tekilleştirilmesi gerekmektedir.
#Örneğin birden fazla şu ifadeden olabilir: USA_ANDROID_MALE_0_18. Bunları groupby'a alıp price ortalamalarını almak gerekmektedir.

#1.yol:
agg_df["customers_level_based"] = [(str(agg_df["COUNTRY"][i]).upper()+"_"+str(agg_df["SOURCE"][i]).upper()+
                                    "_"+str(agg_df["SEX"][i]).upper()+"_"+str(agg_df["AGE_CAT"][i]).upper())
                                   for i in range(len(agg_df.index))]

#Yeni seviye tabanlarını tekilleştirmek için
agg_df = agg_df.groupby("customers_level_based").agg({"PRICE": "mean"}).reset_index()

#2.yol:
agg_df["customers_level_based"] = [
    str(value[0]).upper() + "_" + str(value[1]).upper() + "_" + str(value[2]).upper() + "_" + str(value[5]).upper() for value in
    agg_df[agg_df.columns].values]



#Yeni müşterileri (personaları) segmentlere ayırınız.
#Yeni müşterileri (Örnek: USA_ANDROID_MALE_0_18) PRICE’a göre 4 segmente ayırınız.
#Segmentleri SEGMENT isimlendirmesi ile değişken olarak agg_df’e ekleyiniz.
#Segmentleri betimleyiniz (Segmentlere göre group by yapıp price mean, max, sum’larını alınız).

agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"],4,labels=["D","C","B","A"])
agg_df.groupby("SEGMENT").agg({"PRICE": ["mean","max","sum"]})


#Yeni gelen müşterileri sınıflandırıp, ne kadar gelir getirebileceklerini tahmin ediniz.
#33 yaşında ANDROID kullanan bir Türk kadını hangi segmente aittir ve ortalama ne kadar gelir kazandırması beklenir?
new user = "TUR_ANDROID_FEMALE_31_40"
#35 yaşında IOS kullanan bir Fransız kadını hangi segmente aittir ve ortalama ne kadar gelir kazandırması beklenir?
new user = "FRA_IOS_FEMALE_31_40"

#1.yol:
new_user = "TUR_ANDROID_FEMALE_31_40"
agg_df[agg_df["customers_level_based"] == new_user]

#2.yol: uzun olan.
def yeni_musteri_geliri_tahminleme(new_user):
    if new_user in agg_df["customers_level_based"].values:
        print(agg_df[agg_df["customers_level_based"] == new_user]["SEGMENT"])
        print(agg_df[agg_df["customers_level_based"] == new_user]["PRICE"])

yeni_musteri_geliri_tahminleme("TUR_ANDROID_FEMALE_31_40")