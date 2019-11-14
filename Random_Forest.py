import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

year = list()
month = list()
day = list()
From = list()
To = list()

train_data = pd.read_csv('Alibaba/train.csv', sep=',', usecols=[1, 2, 3, 4, 5, 7], nrows=10000)
train_label = train_data.groupby(['Log_Date', 'FROM', 'TO', 'Price']).size()
values = train_label.values
label = train_label.index
for i in range(len(train_label.index.values)):
    year.append(int(train_label.index.values[i][0][0:4]))
    month.append(int(train_label.index.values[i][0][5:7]))
    day.append(train_label.index.values[i][0][8:10])
    From.append(train_label.index.values[i][1])
    To.append(train_label.index.values[i][2])

df = pd.DataFrame({
    "year": year,
    'month': month,
    'day': day,
    "From": From,
    "To": To
})
# from sklearn.metrics import accuracy_score
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(df, values, test_size=0.2)

clf = RandomForestClassifier(n_estimators=850)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_tr_pred = clf.predict(X_train)
print("test Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("train Accuracy:", metrics.accuracy_score(y_train, y_tr_pred))
