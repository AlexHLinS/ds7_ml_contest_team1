# Решение для итогового проекта в рамках курса Машинное обучение (DS-7) Команда 1

## 1. Состав команды

- Цветкова Алена [@nihaoalena](https://github.com/nihaoalena)
- Шкиль Алексей [@alexhlins](https://github.com/alexhlins)

## 2. Решаемая задача

Нашей командой была выбрана задача
"Expresso Churn Prediction Challenge".
Суть которой заключается в поиске модели,
позволяющей с наибольшой точностью
предугадать изменения лояльности
клиента, что позволит предупредить
его "уход".

Исходное описание задачи доступны по [ссылке](https://zindi.africa/competitions/expresso-churn-prediction/).
Метрикой оценки для этой задачи является **площадь под кривой (AUC)**.

## 3. Исследование данных

Согласно описанию, представленны обезличенные данные пользователей:

- user_id - индитификатор пользователя
- REGION - местоположение клиента
- TENURE - времени с момента заключения контракта
- MONTANT - сумма пополнения
- FREQUENCE_RECH - количество пополнений счета
- REVENUE - ежемесячный доход с клиента
- ARPU_SEGMENT - приведенный ежемесячные поступления за последние 90 дней (сумма поступлений за 90 дней / 3)
- FREQUENCE - количество пополнений счета
- DATA_VOLUME - количество подключений
- ON_NET - звонков внутри сети
- ORANGE - звонков абонентам ORANGE
- TIGO - звонков абонентам Tigo
- ZONE1 - звонков zones1
- ZONE2 - звонков zones2
- MRG - клиент ушел
- REGULARITY - время активности за последний 90 дней
- TOP_PACK - наиболее часто используемый пакет услуг
- FREQ_TOP_PACK - количество активаций часто используемого пакета услуг
- CHURN - пользователь ушел(целевая переменная)

В процессе исследования, было выявлено:

| № п/п | Акроним поля   | Процент отсустующих данных | Описание наблюдения                                                         |
|-------|----------------|----------------------------|-----------------------------------------------------------------------------|
| 1     | MRG            | -                          | На всем объеме обучающих данных имеет единственное значение                 |
| 2     | REGION         | ~39%                       | Возможно что заполнение каким-либо средним приведет к искажению результатов |
| 3     | MONTANT        | ~35%                       | Аналогично предыдущему                                                      |
| 4     | FREQUENCE_RECH | ~35%                       | Аналогично предыдущему                                                      |
| 5     | REVENUE        | ~34%                       | Аналогично предыдущему                                                      |
| 6     | ARPU_SEGMENT   | ~34%                       | Аналогично предыдущему                                                      |
| 7     | FREQUENCE      | ~34%                       | Аналогично предыдущему                                                      |
| 8     | DATA_VOLUME    | ~49%                       | Трактуем Nan как отсутствие активности                                      |
| 9     | ON_NET         | ~37%                       | Аналогично предыдущему                                                      |
| 10    | ORANGE         | ~42%                       | Аналогично предыдущему                                                      |
| 11    | TIGO           | ~60%                       | Аналогично предыдущему                                                      |
| 12    | ZONE1          | ~92%                       | Аналогично предыдущему                                                      |
| 13    | ZONE2          | ~94%                       | Аналогично предыдущему                                                      |
| 14    | TOP_PACK       | ~42%                       | Возможно что заполнение каким-либо средним приведет к искажению результатов | 
| 15    | FREQ_TOP_PACK  | ~42%                       | Возможно что заполнение каким-либо средним приведет к искажению результатов |

## 4. Baseline

В задании был представлен [Jupyter notebook](StarterNotebook.ipynb) с обзором предоставляемого набора данных, а также
предложена модель решения задачи.

В качестве Baseline предложен алгоритм RandomForest,
перед применением которого были произведены следующие операции:

- удалены колонки REGION, TOP_PACK и MRG;
- заполнение пропущенных данных медианными значениями;
- данных с использованием StandardScaler.

## 5. Что было улучшено

### 5.1 Первая итерация

При первом приближении было принято решение изменить подход к исключению
данных из набора, а также заполнения пропущенных значений:

| № п/п | Акроним поля   | Принятое решение                                                               |
|-------|----------------|--------------------------------------------------------------------------------|
| 1     | MRG            | Удалить из набора данных                                                       |
| 2     | REGION         | Заполнить признаком “OTHER”                                                    |
| 3     | MONTANT        | Заполнить значением 0                                                          |
| 4     | FREQUENCE_RECH | Заполнить значением 0                                                          |
| 5     | REVENUE        | Заполнить значением 0                                                          |
| 6     | ARPU_SEGMENT   | Заполнить значением 0                                                          |
| 7     | FREQUENCE      | Заполнить значением 0                                                          |
| 8     | DATA_VOLUME    | Заполнить значением 0                                                          |
| 9     | ON_NET         | Заполнить значением 0                                                          |
| 10    | ORANGE         | Заполнить значением 0                                                          |
| 11    | TIGO           | Заполнить значением 0                                                          |
| 12    | ZONE1          | Заполнить значением 0                                                          |
| 13    | ZONE2          | Заполнить значением 0                                                          |
| 14    | TOP_PACK       | Заполнить значением 0                                                          |
| 15    | FREQ_TOP_PACK  | Удалить из набора данных |

### 5.2 Вторая итерация

При первом приближении было принято решение изменить подход к исключению
данных из набора, а также заполнения пропущенных значений:

| № п/п | Акроним поля   | Принятое решение                                                               |
|-------|----------------|--------------------------------------------------------------------------------|
| 1     | MRG            | Удалить из набора данных                                                       |
| 2     | REGION         | Заполнить признаком “OTHER”                                                    |
| 3     | MONTANT        | Заполнить значением 0                                                          |
| 4     | FREQUENCE_RECH | Заполнить значением 0                                                          |
| 5     | REVENUE        | Заполнить значением 0                                                          |
| 6     | ARPU_SEGMENT   | Заполнить значением 0                                                          |
| 7     | FREQUENCE      | Заполнить значением 0                                                          |
| 8     | DATA_VOLUME    | Заполнить значением 0                                                          |
| 9     | ON_NET         | Заполнить значением 0                                                          |
| 10    | ORANGE         | Заполнить значением 0                                                          |
| 11    | TIGO           | Заполнить значением 0                                                          |
| 12    | ZONE1          | Заполнить значением 0                                                          |
| 13    | ZONE2          | Заполнить значением 0                                                          |
| 14    | TOP_PACK       | С помощью кластеризации определить возможное значение                          |
| 15    | FREQ_TOP_PACK  | После заполнение TOP_PACK с помощью кластеризации определим возможное значение |

## 6. Описание выбранной модели

## 7. Сравнительная таблица использованных моделей
Значения метрик, полученных при локальном тестировании на предоставленных данных:

| № | Название модели                  | Описание модели | Предобработка данных|roc-auc на валидации  | roc-auc на leaderboard | 
|---|----------------------------------|-----------------|---------------------|----------------------|------------------------|
| 1 | RandomForestClassifier(BaseLine) | bootstrap=True, criterion = "gini", max_depth=7, n_estimators=200, random_state=1, verbose=True| ....................|....................| 0.500108 |


## 8. Итоговые метрики

