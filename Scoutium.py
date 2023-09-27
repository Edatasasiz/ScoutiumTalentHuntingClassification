####################
# İş Problemi: Scout’lar tarafından izlenen futbolcuların özelliklerine verilen puanlara göre, oyuncuların hangi sınıf
# (average, highlighted) oyuncu olduğunu tahminleme.
####################


#####################################
# Veri Seti Hikayesi : Veri seti Scoutium’dan maçlarda gözlemlenen futbolcuların özelliklerine göre
# scoutların değerlendirdikleri futbolcuların, maç içerisinde puanlanan özellikleri ve puanlarını içeren
# bilgilerden oluşmaktadır.

# scoutium_attributes.csv

# task_response_id : Bir scoutun bir maçta bir takımın kadrosundaki tüm oyunculara dair değerlendirmelerinin kümesi
# match_id : İlgili maçın id'si
# evaluator_id : Değerlendiricinin(scout'un) id'si
# player_id : İlgili oyuncunun id'si
# position_id : İlgili oyuncunun o maçta oynadığı pozisyonun id’si
# 1: Kaleci
# 2: Stoper
# 3: Sağ bek
# 4: Sol bek
# 5: Defansif orta saha
# 6: Merkez orta saha
# 7: Sağ kanat
# 8: Sol kanat
# 9: Ofansif orta saha
# 10: Forvet
# analysis_id : Bir scoutun bir maçta bir oyuncuya dair özellik değerlendirmelerini içeren küme
# attribute_id : Oyuncuların değerlendirildiği her bir özelliğin id'si
# attribute_value : Bir scoutun bir oyuncunun bir özelliğine verdiği değer(puan)


# scoutium_potential_labels.csv

# task_response_id : Bir scoutun bir maçta bir takımın kadrosundaki tüm oyunculara dair değerlendirmelerinin kümesi
# match_id : İlgili maçın id'si
# evaluator_id : Değerlendiricinin(scout'un) id'si
# player_id : İlgili oyuncunun id'si
# potential_label : Bir scoutun bir maçta bir oyuncuyla ilgili nihai kararını belirten etiket. (hedef değişken)

##################################

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

# ########## KNN ##########
# Accuracy: 0.8522
# Auc: 0.7663
# Recall: 0.3233
# Precision: 0.8333
# F1: 0.4463
# ########## CART ##########
# Accuracy: 0.8192
# Auc: 0.7361
# Recall: 0.5933
# Precision: 0.5708
# F1: 0.5694
# ########## RF ##########
# Accuracy: 0.8782
# Auc: 0.9133
# Recall: 0.47
# Precision: 0.9167
# F1: 0.5956
# ########## GBM ##########
# Accuracy: 0.8524
# Auc: 0.8895
# Recall: 0.5233
# Precision: 0.705
# F1: 0.587
# ########## XGBoost ##########
# Accuracy: 0.8672
# Auc: 0.8771
# Recall: 0.61
# Precision: 0.7608
# F1: 0.6447
# ########## CatBoost ##########
# Accuracy: 0.8856
# Auc: 0.9015
# Recall: 0.47
# Precision: 0.9667
# F1: 0.6088
# ########## LightGBM ##########
# Accuracy: 0.8856
# Auc: 0.889
# Recall: 0.5767
# Precision: 0.8464
# F1: 0.6605



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




















knn_params = {"n_neighbors": range(2, 50)}

cart_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2, 30)}

rf_params = {"max_depth": [8, 15, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [15, 20],
             "n_estimators": [200, 300]}

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 200],
                  "colsample_bytree": [0.5, 1]}

lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [300, 500],
                   "colsample_bytree": [0.7, 1]}

classifiers = [('KNN', KNeighborsClassifier(), knn_params),
               ("CART", DecisionTreeClassifier(), cart_params),
               ("RF", RandomForestClassifier(), rf_params),
               ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgboost_params),
               ('LightGBM', LGBMClassifier(), lightgbm_params)]


def hyperparameter_optimization(X, y, cv=3, scoring="roc_auc"):
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models

best_models = hyperparameter_optimization(X, y)

# ########## RF ##########
# roc_auc (Before): 0.8884
# roc_auc (After): 0.8906
# RF best params: {'max_depth': 8, 'max_features': 'auto', 'min_samples_split': 15, 'n_estimators': 200}
