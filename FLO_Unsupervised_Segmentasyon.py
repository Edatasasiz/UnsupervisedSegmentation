import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
import datetime as dt
pd.set_option("display.max_columns", None)
pd.set_option('display.width', 500)


################################ Görev 1: Veriyi Hazırlama  ##########################################

################
# Adım 1: flo_data_20K.csv verisini okutunuz.
###############

df = pd.read_csv("Machine_Learning/Case3_FLO_Unsupervised/flo_data_20k.csv")
df.head()
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
check_df(df)


for col in df.columns:
    if "date" in col:
        df[col] = pd.to_datetime(df[col])

df.info()


################
# Adım 2: Müşterileri segmentlerken kullanacağınız değişkenleri seçiniz.
#         Not: Tenure (Müşterinin yaşı), Recency (en son kaç gün önce alışveriş yaptığı) gibi
#               yeni değişkenler oluşturabilirsiniz.
###############

df["last_order_date"].max()     # 2021-05-30
today_date = dt.datetime(2021, 6, 1)    # analiz 2 gün sonra

df["total_order"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["total_value"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]

df["tenure"] = (today_date - df["first_order_date"]).dt.days
df["recency"] = (today_date - df["last_order_date"]).dt.days
df["frequency"] = df["total_order"]
df["monetary"] = df["total_value"]
df.head()

df_1 = df[['tenure','monetary', 'frequency', 'recency']]
df_1.head()



######################## Görev 2: K-Means ile Müşteri Segmentasyonu  ###############################

################
# Adım 1: Değişkenleri standartlaştırınız.
###############

sc = MinMaxScaler((0, 1))
df_scaled = sc.fit_transform(df_1)
df_scaled[0:5]


################
# Adım 2: Optimum küme sayısını belirleyiniz.
###############

kmeans = KMeans()
ssd = []
K = range(1, 30)

for k in K:
    kmeans = KMeans(n_clusters=k).fit(df_1)
    ssd.append(kmeans.inertia_)

plt.plot(K, ssd, "bx-")
plt.xlabel("Farklı K Değerlerine Karşılık SSE/SSR/SSD")
plt.title("Optimum Küme sayısı için Elbow Yöntemi")
plt.show()


kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(df_scaled)
elbow.show()

elbow.elbow_value_
# 6


################
# Adım 3: Modelinizi oluşturunuz ve müşterilerinizi segmentleyiniz.
###############

kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(df_scaled)

clusters_kmeans = kmeans.labels_
df["cluster"] = clusters_kmeans
df["cluster"] = df["cluster"] + 1

df.head()

################
# Adım 4: Her bir segmenti istatistiksel olarak inceleyiniz.
###############

df.groupby("cluster").agg(["count","mean","median"])




############### Görev 3: Hierarchical Clustering ile Müşteri Segmentasyonu  #####################

################
# Adım 1: Görev 2'de standartlaştırdığınız dataframe'i kullanarak optimum küme sayısını belirleyiniz.
###############

hc_average = linkage(df_scaled, "average")

plt.figure(figsize=(7, 5))
plt.title("Dendrograms")
dendrogram(hc_average,
           truncate_mode="lastp",
           p=10,
           show_contracted=True,
           leaf_font_size=10)
plt.axhline(y=0.5, color='r', linestyle='--')
plt.axhline(y=0.6, color='b', linestyle='--')
plt.show()



################
# Adım 2: Modelinizi oluşturunuz ve müşterilerinizi segmentleyiniz.
###############

from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=5, linkage="average")
clusters = cluster.fit_predict(df_scaled)

df["hi_cluster_no"] = clusters
df["hi_cluster_no"] = df["hi_cluster_no"] + 1

df.head()



################
# Adım 3: Her bir segmenti istatistiksel olarak inceleyeniz.
###############

df.groupby("hi_cluster_no")["recency","frequency","monetary"].agg(["count","mean","median"])


