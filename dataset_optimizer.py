import pickle

import pandas as pd
import numpy as np


# preparing data for catboost
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split


def modify_X(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe.drop(columns=['MRG'])
    dataframe["REGION"] = dataframe["REGION"].fillna("OTHER")
    dataframe["MONTANT"] = dataframe["MONTANT"].fillna(0)
    dataframe["FREQUENCE_RECH"] = dataframe["FREQUENCE_RECH"].fillna(0)
    dataframe["REVENUE"] = dataframe["REVENUE"].fillna(0)
    dataframe["ARPU_SEGMENT"] = dataframe["ARPU_SEGMENT"].fillna(0)
    dataframe["FREQUENCE"] = dataframe["FREQUENCE"].fillna(0)
    dataframe["DATA_VOLUME"] = dataframe["DATA_VOLUME"].fillna(0)
    dataframe["ON_NET"] = dataframe["ON_NET"].fillna(0)
    dataframe["ORANGE"] = dataframe["ORANGE"].fillna(0)
    dataframe["TIGO"] = dataframe["TIGO"].fillna(0)
    dataframe["ZONE1"] = dataframe["ZONE1"].fillna(0)
    dataframe["ZONE2"] = dataframe["ZONE2"].fillna(0)
    # оптимизируем типы
    try:
        dataframe['CHURN'] = dataframe['CHURN'].astype(bool)
    except KeyError:
        pass

    dataframe['MONTANT'] = dataframe['MONTANT'].astype(np.int32)
    dataframe['FREQUENCE_RECH'] = dataframe['FREQUENCE_RECH'].astype(np.int16)
    dataframe['REVENUE'] = dataframe['REVENUE'].astype(np.int32)
    dataframe['ARPU_SEGMENT'] = dataframe['ARPU_SEGMENT'].astype(np.int32)
    dataframe['FREQUENCE'] = dataframe['FREQUENCE'].astype(np.int16)
    dataframe['DATA_VOLUME'] = dataframe['DATA_VOLUME'].astype(np.int32)
    dataframe['ON_NET'] = dataframe['ON_NET'].astype(np.int32)
    dataframe['ORANGE'] = dataframe['ORANGE'].astype(np.int16)
    dataframe['TIGO'] = dataframe['TIGO'].astype(np.int16)
    dataframe['ZONE1'] = dataframe['ZONE1'].astype(np.int16)
    dataframe['ZONE2'] = dataframe['ZONE2'].astype(np.int16)
    dataframe['REGULARITY'] = dataframe['REGULARITY'].astype(np.int16)
    return dataframe


test_df = pd.read_csv('Test.csv')
train_df = pd.read_csv('Train.csv')

train_df = modify_X(train_df)
test_df = modify_X(test_df)

# выделяем набор данных для предсказания TOP_PACK
top_pack_predict_df = train_df[train_df["TOP_PACK"].isna()].drop(columns=['TOP_PACK', 'FREQ_TOP_PACK','CHURN', 'user_id'])
# набор данных для обучения и теста метода KNN для дальнейшего предсказания TOP_PACK
# разделение на параметры и целевую переменную
top_pack_traintest_df = train_df[~train_df["TOP_PACK"].isna()]
top_pack_X_df = top_pack_traintest_df.drop(columns=['TOP_PACK', 'FREQ_TOP_PACK','CHURN','user_id'])
top_pack_y_df = top_pack_traintest_df['TOP_PACK']

# создаем модель для предсказания TOP_PACK методом кластеризации,
# количество кластеров == кол-ву возможных значений TOP_PACK
cluster_model1 = KMeans(n_clusters=len(train_df['TOP_PACK'].unique()))

top_pack_X_df_train, top_pack_X_df_test, top_pack_y_df_train, top_pack_y_df_test = train_test_split(top_pack_X_df,
                                                                                                    top_pack_y_df,
                                                                                                    train_size=0.7,
                                                                                                    random_state=777)
# проверяем если модельку уже делали, то просто загружаем её, если нет - обучаем
try:
    cluster_model1 = pickle.load(open('TOP_PACK_predict_model.pkl', 'rb'))
except FileNotFoundError:
    print('File not found, training new model')
    cluster_model1.fit(top_pack_X_df_train)

predicted_y_train = cluster_model1.predict(top_pack_X_df_train)
predicted_y_test = cluster_model1.predict(top_pack_X_df_test)

print(
    f"Результаты кластеризации:\nТренировочная выборка: {rand_score(top_pack_y_df_train, predicted_y_train)}\nТестовая выборка: {rand_score(top_pack_y_df_test, predicted_y_test)}")




train_df.to_csv('train_adapted.csv')
test_df.to_csv('test_adapted.csv')
