import numpy as np
import pandas as pd
import tensorflow as tf
import jdatetime
import keras
# load data

train_data = pd.read_csv('Alibaba/train.csv', usecols=[1, 3, 4])
date = list()
From = list()
To = list()
train_label = train_data.groupby(['Log_Date', 'FROM', 'TO']).size()
label = list()
for i in range(len(train_label.index.values)):
    label.append(train_label[i])

# select log_date and FROM and TO

for i in range(len(train_label.index.values)):
    date.append(train_label.index.values[i][0])
    From.append(train_label.index.values[i][1])
    To.append(train_label.index.values[i][2])

d = np.vstack((date, From, To)).T

np.savetxt('train.csv', d, header='date,From,to', comments='', delimiter=',', fmt='%s')
train_label = np.array(label)
train_data = pd.read_csv('train.csv').values
test_data = pd.read_csv('Alibaba/test.csv').values
test_data_title = pd.read_csv('Alibaba/test.csv', usecols=[0, 1, 2])
last = test_data_title.groupby(['Log_Date', 'From', 'To']).size()
date.clear()
From.clear()
To.clear()


# preprocess the train data
def train_process():
    global train_data, train_label
    i = 0
    temp = np.zeros((79990, 6))
    for item in train_data[:79990]:
        date = jdatetime.datetime.strptime(item[0], "%Y/%m/%d")
        day = date.day
        month = date.strftime("%m")
        dayofweek = date.weekday()
        season = (int(month) % 12 + 3) // 3

        temp[i][0] = day
        temp[i][1] = int(month)
        temp[i][2] = dayofweek
        temp[i][3] = season
        temp[i][4] = item[1]
        temp[i][5] = item[2]

        i += 1

    temp = temp.astype('float32')
    train_label = train_label.astype('float32')

    day = keras.utils.to_categorical(temp[:, 0], 32)
    month = keras.utils.to_categorical(temp[0:, 1], 13)
    dayofweek = keras.utils.to_categorical(temp[1:, 2], 8)
    season = keras.utils.to_categorical(temp[2:, 3], 5)
    source = keras.utils.to_categorical(temp[3:, 4])
    destination = keras.utils.to_categorical(temp[4:, 5])

    train_label = keras.utils.to_categorical(train_label)

    dataset = pd.concat(
        [pd.DataFrame(data=day), pd.DataFrame(data=month), pd.DataFrame(data=dayofweek), pd.DataFrame(data=season),
         pd.DataFrame(data=source), pd.DataFrame(data=destination)], axis=1)

    train_data = dataset.values
    return train_data, train_label


# preprocess test data
def test_process():
    global test_data
    i = 0
    temp = np.zeros((34676, 6))
    for item in test_data[:34676]:
        date = jdatetime.datetime.strptime(item[0], "%Y/%m/%d")
        day = date.day
        month = date.strftime("%m")
        dayofweek = date.weekday()
        season = (int(month) % 12 + 3) // 3

        temp[i][0] = day
        temp[i][1] = int(month)
        temp[i][2] = dayofweek
        temp[i][3] = season
        temp[i][4] = item[1]
        temp[i][5] = item[2]

        i += 1

    temp = temp.astype('float32')

    day = keras.utils.to_categorical(temp[:, 0], 32)
    month = keras.utils.to_categorical(temp[0:, 1], 13)
    weekday = keras.utils.to_categorical(temp[1:, 2], 8)
    season = keras.utils.to_categorical(temp[2:, 3], 5)
    source = keras.utils.to_categorical(temp[3:, 4])
    destination = keras.utils.to_categorical(temp[4:, 5])

    dataset = pd.concat(
        [pd.DataFrame(data=day), pd.DataFrame(data=month), pd.DataFrame(data=weekday), pd.DataFrame(data=season),
         pd.DataFrame(data=source), pd.DataFrame(data=destination)], axis=1)

    test_data = dataset.values
    return test_data


# create model

train_data, train_label = train_process()
test_data = test_process()

model = keras.models.Sequential()
model.add(keras.layers.Dense(20, activation=tf.nn.relu, input_shape=(212,)))
model.add(keras.layers.Dense(35, activation=tf.nn.relu))
model.add(keras.layers.Dense(45, activation=tf.nn.relu))
model.add(keras.layers.Dense(340, activation=tf.nn.softmax))

model.compile(loss='categorical_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])

# fit model

model.fit(train_data[:70000], train_label[:70000], epochs=2)

# evaluate model
value_loss, value_acc = model.evaluate(train_data[70000:], train_label[70000:])

# print(value_loss, value_acc)
model.save('model.model')
new_model = keras.models.load_model('model.model')
tt = train_label[70000:]
ttt = train_data[70000:]
predict = new_model.predict(ttt)

x = 0
for i in range(len(tt)):
    print(predict[0])
    if np.argmax(tt[i]) != np.argmax(predict[i]):
        x += 1
print('correct predict number:')
print(len(tt) - x)
Sales = list()
predict = new_model.predict(test_data)
for i in range(len(test_data)):
    Sales.append(np.argmax(predict[i]))

# save data

for i in range(len(last.index.values)):
    date.append(last.index.values[i][0])
    From.append(last.index.values[i][1])
    To.append(last.index.values[i][2])

d = np.vstack((date, From, To, Sales)).T
np.savetxt('last.csv', d, header='Log_Date,FROM,TO,Sales', comments='', delimiter=',', fmt='%s')
