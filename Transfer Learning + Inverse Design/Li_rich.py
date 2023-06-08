import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from keras.layers import Flatten, Dropout, Activation, Dense
from keras.layers.convolutional import MaxPooling1D, Conv1D
from keras.models import Sequential
from sklearn.utils import check_array
from keras.callbacks import ModelCheckpoint
from matplotlib.pyplot import figure
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import KFold
from itertools import zip_longest
from keras.models import load_model
import csv
import random
import tensorflow as tf
import os

rand_state = 42

os.environ['PYTHONHASHSEED']=str(rand_state)
tf.random.set_seed(rand_state)
np.random.seed(rand_state)
random.seed(rand_state)

#import data
data_import = pd.read_csv('./impute_Li_NCM_MICE.csv')
Datshape = data_import.shape
total_parameter = len(data_import)

from sklearn.model_selection import train_test_split
features = ['Li', 'Ni ', 'Co', 'Mn',  'sin_temp', 'sin_t', 'm_temp', 'cut_off', 'c_rate']
label = ['Dis_cap']
X = data_import.loc[:,features].values #data_import[features].values
Y = data_import.loc[:,label].values #data_import[label].values
perc_test = 0.2


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=perc_test, random_state=None)
print('shape', Y_train.shape)
Train_data = (len(X_train) / len(data_import.index))
Test_data = (len(X_test) / len(data_import.index))
print("{0:0.2f} % for training dataset".format(Train_data * 100))
print("{0:0.2f} % for testing dataset".format(Test_data * 100))


scaler = StandardScaler()
scaler.fit(X_train)
X_scaled_train = scaler.transform(X_train)
X_scaled_test = scaler.transform(X_test)
scaler2 = StandardScaler()
scaler2.fit(Y_train)
Y_scaled_train = scaler2.transform(Y_train)
Y_scaled_test = scaler2.transform(Y_test)

X_scaled_train = np.expand_dims(X_scaled_train, axis=2)
X_scaled_test = np.expand_dims(X_scaled_test, axis=2)

nSNP = X_scaled_train.shape[1]
nStride = 3

model = Sequential()
model.add(Conv1D(256, kernel_size=3, input_shape=(nSNP,1), activation='tanh'))
model.add(Conv1D(128, kernel_size=3, activation='tanh'))
model.add(Flatten())
model.add(Dense(128, activation='tanh'))
model.add(Dense(64, activation='tanh'))
model.add(Dense(32, activation = 'tanh'))
model.add(Dropout(0.3))
model.add(Dense(1))
model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])
path = "./_model.h5"
checkpoint = ModelCheckpoint(path, monitor='loss', save_best_only=True, mode='min', verbose=1)
model.summary()
# fit model
history = model.fit(X_scaled_train, Y_scaled_train, validation_data=(X_scaled_test, Y_scaled_test), epochs=500, verbose=1,  callbacks=[checkpoint])

train_acu = history.history['accuracy']
val_acu = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']

########### plot performance ##############
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

d = [train_acu, val_acu, train_loss, val_loss]
hist_eval = zip_longest(*d, fillvalue = '')
headers = ["train accuracy", 'eval accuracy', 'train loss', 'eval loss']
with open("History_Li_no_TL.csv", "w", newline='') as file2:
    writer = csv.writer(file2)
    writer.writerow(headers)
    writer.writerows(hist_eval)

predicted_Y_train = model.predict(X_scaled_train)
predicted_Y_test = model.predict(X_scaled_test)

predicted_Y_train = scaler2.inverse_transform(predicted_Y_train)
predicted_Y_test = scaler2.inverse_transform(predicted_Y_test)


Y_train = scaler2.inverse_transform(Y_scaled_train)
Y_test = scaler2.inverse_transform(Y_scaled_test)
Y_train = Y_train.squeeze()
Y_test = Y_test.squeeze()

dii = [Y_train, predicted_Y_train, Y_test, predicted_Y_test]
result_all = zip_longest(*dii, fillvalue = '')
headers = ["train", 'train_predict', 'test', 'test_predict']
with open("RESULT_Li_no_TL.csv", "w", newline='') as file3:
    writer = csv.writer(file3)
    writer.writerow(headers)
    writer.writerows(result_all)

y_predict = predicted_Y_train
# plt.xlim(100, 300)
# plt.ylim(100, 300)
plt.scatter(Y_train, y_predict)
plt.title('$R^2$: {0:.3f}'.format(r2_score(Y_train, y_predict)))
plt.ylabel('Predicted Capacity (mAh/g)')
plt.xlabel('Actual Capacity (mAh/g)')

y_predict_V = predicted_Y_test
# plt.xlim(100, 300)
# plt.ylim(100, 300)
plt.scatter(Y_test, y_predict_V)
plt.title('$R^2$: {0:.3f}'.format(r2_score(Y_test, y_predict_V)))
plt.ylabel('Predicted Validation Capacity (mAh/g)')
plt.xlabel('Actual Validation Capacity (mAh/g)')
plt.show()

