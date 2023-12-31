import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler


####################
# Adım 1: scoutium_attributes.csv ve scoutium_potential_labels.csv dosyalarını okutunuz.
###################

df_att = pd.read_csv("Machine_Learning/Case2_Scoutium/scoutium_attributes.csv", sep=';')
df_pot = pd.read_csv("Machine_Learning/Case2_Scoutium/scoutium_potential_labels.csv", sep=';')

df_att.head()
df_att.shape        # (10730, 8)

df_pot.head()
df_pot.shape        # (322, 5)


######################
# Adım 2: Okutmuş olduğumuz csv dosyalarını merge fonksiyonunu kullanarak birleştiriniz.
# ("task_response_id", 'match_id', 'evaluator_id', "player_id" 4 adet değişken üzerinden birleştirme işlemini
# gerçekleştiriniz.)
######################

df = pd.merge(df_att, df_pot, on=["task_response_id", 'match_id', 'evaluator_id', "player_id"])
df.head()



#######################
# Adım 3: position_id içerisindeki Kaleci (1) sınıfını veri setinden kaldırınız.
#######################

df.drop(df[df["position_id"] == 1].index, inplace=True)
df["position_id"].value_counts()



#######################
# Adım 4: : potential_label içerisindeki below_average sınıfını veri setinden kaldırınız.
#           ( below_average sınıfı tüm verisetinin %1'ini oluşturur)
#######################

df["potential_label"].value_counts()
df.drop(df[df["potential_label"] == "below_average"].index , inplace=True)



#######################
# Adım 5: : Oluşturduğunuz veri setinden “pivot_table” fonksiyonunu kullanarak bir tablo oluşturunuz.
#           Bu pivot table'da her satırda bir oyuncu olacak şekilde manipülasyon yapınız.
#######################

# Adım 1: : İndekste “player_id”,“position_id” ve “potential_label”, sütunlarda “attribute_id” ve
# değerlerde scout’ların oyunculara verdiği puan “attribute_value” olacak şekilde pivot table’ı oluşturunuz.

df_table = pd.pivot_table(df, index = ["player_id", "position_id", "potential_label"],
                              columns="attribute_id",
                              values="attribute_value").reset_index()

# Adım 2 : “reset_index” fonksiyonunu kullanarak indeksleri değişken olarak atayınız ve
#          “attribute_id” sütunlarının isimlerini stringe çeviriniz.

df_table.columns = df_table.columns.astype(str)
df_table.head()

df_table.info()



#######################
# Adım 6: : Label Encoder fonksiyonunu kullanarak “potential_label” kategorilerini
#           (average, highlighted) sayısal olarak ifade ediniz.
#######################

df_table["potential_label"] = LabelEncoder().fit_transform(df_table["potential_label"])
df_table.head()



#######################
# Adım 7: : Sayısal değişken kolonlarını “num_cols” adıyla bir listeye atayınız.
#######################

df_table.info()

df["position_id"].value_counts()
df["potential_label"].value_counts()

num_cols = [col for col in df_table.columns if col not in ["player_id", "position_id", "potential_label"]]
df_table_num = df_table[num_cols]



#######################
# Adım 8 : Kaydettiğiniz bütün “num_cols” değişkenlerindeki veriyi ölçeklendirmek için StandardScaler uygulayınız.
#######################

X_scaled = StandardScaler().fit_transform(df_table[num_cols])
df_table[num_cols] = pd.DataFrame(X_scaled, columns=df_table[num_cols].columns)

df_table[num_cols].head()



#######################
# Adım 9 : Elimizdeki veri seti üzerinden minimum hata ile futbolcuların potansiyel etiketlerini tahmin eden bir
#          makine öğrenmesi modeli geliştiriniz. (Roc_auc, f1, precision, recall, accuracy metriklerini yazdırınız.)
#######################

X = df_table.drop(["potential_label", "player_id", "position_id"], axis=1)
y = df_table["potential_label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = [('LR', LogisticRegression()),
          ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier()),
          ('RF', RandomForestClassifier()),
          ('GBM', GradientBoostingClassifier()),
          ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
          ("CatBoost", CatBoostClassifier(verbose=False)),
          ("LightGBM", LGBMClassifier())]

for name, model in models:
    cv_results = cross_validate(model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
    print(f"########## {name} ##########")
    print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
    print(f"Auc: {round(cv_results['test_roc_auc'].mean(), 4)}")
    print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
    print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
    print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")



#######################
# Adım 10 : Değişkenlerin önem düzeyini belirten feature_importance fonksiyonunu kullanarak
#           özelliklerin sıralamasını çizdiriniz.
#######################

def plot_importance(model, features, num=len(X), save=False):

    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title("Features")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("importances.png")

model = LGBMClassifier()
model.fit(X, y)

plot_importance(model, X)
