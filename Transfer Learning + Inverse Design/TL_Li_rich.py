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
from keras.models import load_model
from itertools import zip_longest
import random as rd
import csv
import random
random.seed(30)
sns.set()

#import data
data_import = pd.read_csv('./impute_Li_NCM_MICE.csv')
Datshape = data_import.shape
total_parameter = len(data_import)
# generate new parameters
lb = [0.01, 0, 0, 0, 1, 0.1, 550, 12, 4.0, 20, 0.05]
ub = [0.3, 0.6, 0.6, 0.8, 30, 2, 1000, 24, 4.6, 30, 2]
offset = 0.2

from sklearn.model_selection import train_test_split
features = ['Li', 'Ni ', 'Co', 'Mn', 'sin_temp', 'sin_t', 'm_temp', 'cut_off', 'c_rate']
label = ['Dis_cap']
X = data_import.loc[:,features].values #data_import[features].values
Y = data_import.loc[:,label].values #data_import[label].values
perc_test = 0.2


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=perc_test, random_state=42)
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
#model_TL.add(Conv1D(64, kernel_size=3, input_shape=(nSNP,1)))
nSNP = X_scaled_train.shape[1]

model = load_model('NCM_model.h5')
model_TL = Sequential()
for i in range(4):
    #i = i+1
    layer = model.layers[i]
    layer.trainable = False
    model_TL.add(layer)
model_TL.add(Dense(8, activation = 'relu', name='TL_layer1'))
model_TL.add(Dropout(0.05))
model_TL.add(Dense(Y_scaled_train.shape[1], activation='tanh', name='TL_layer2'))
model_TL.compile(loss='mse', optimizer='adam', metrics=['mean_squared_error']) #change the metric to mean_squared_error (chi hao 3Aug2021)
path = "./Li_model.h5"
model_TL.summary()
checkpoint = ModelCheckpoint(path, monitor='loss', save_best_only=True, mode='min', verbose=1)

# fit model
history = model_TL.fit(X_scaled_train, Y_scaled_train, validation_data=(X_scaled_test, Y_scaled_test), epochs=2000, verbose=1,  callbacks=[checkpoint])

train_acu = history.history['acc']
val_acu = history.history['val_acc']
train_loss = history.history['loss']
val_loss = history.history['val_loss']
d = [train_acu, val_acu, train_loss, val_loss]
hist_eval = zip_longest(*d, fillvalue = '')
headers = ["train accuracy", 'eval accuracy', 'train loss', 'eval loss']
with open("History_Li_rich.csv", "w", newline='') as file2:
    writer = csv.writer(file2)
    writer.writerow(headers)
    writer.writerows(hist_eval)

predicted_Y_train = model_TL.predict(X_scaled_train)
predicted_Y_test = model_TL.predict(X_scaled_test)

predicted_Y_train = scaler2.inverse_transform(predicted_Y_train)
predicted_Y_test = scaler2.inverse_transform(predicted_Y_test)


Y_train = scaler2.inverse_transform(Y_scaled_train)
Y_test = scaler2.inverse_transform(Y_scaled_test)
Y_train = Y_train.squeeze()
Y_test = Y_test.squeeze()
# for y in range(len(Y_train)):
#    with open('D:/python_machine_learning/actual_svm.dat', 'w') as text2:
#        text2.writelines("%s\n" % place for place in Y_train)

dii = [Y_train, predicted_Y_train, Y_test, predicted_Y_test]
result_all = zip_longest(*dii, fillvalue = '')
headers = ["train", 'train_predict', 'test', 'test_predict']
with open("RESULT_Li.csv", "w", newline='') as file3:
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

for i in range(1000):
    p1 = round(rd.uniform(lb[0], ub[0]), 3)
    p2 = round(rd.uniform(lb[1], ub[1]), 3)
    p3 = round((1 - (p1 + p2)) * offset, 3)
    p4 = round(1 - (p1 + p2 + p3), 3)
    p5 = round(rd.uniform(lb[4], ub[4]), 2)
    p6 = round(rd.uniform(lb[5], ub[5]), 2)
    p7 = round(rd.uniform(lb[6], ub[6]), 2)
    p8 = round(rd.uniform(lb[7], ub[7]), 2)
    p9 = round(rd.uniform(lb[8], ub[8]), 2)
    p10 = round(rd.uniform(lb[9], ub[9]), 2)
    p11 = round(rd.uniform(lb[10], ub[10]), 2)
    new_param_ = np.hstack((p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11)).reshape(1, -1)
    #print(new_param_)
    new_param_Scale = scaler.transform (new_param_)
    new_param_Scale = np.expand_dims(new_param_Scale, axis=2)
    y_predict = model_TL.predict(new_param_Scale)
    y_predict = scaler2.inverse_transform(y_predict).round()
    pred_y = " ".join(map(str, y_predict)).replace('[', ' ').replace(']', ' ').replace("'", ' ').replace(".", ' ')
    pred_y = [xi for xi in pred_y.split()]

    with open('NEW_prediction.csv', 'a', newline='') as csvfile:
        fieldnames = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11', 'pred_y']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writerow({'p1': p1, 'p2': p2, 'p3': p3, 'p4': p4, 'p5': p5, 'p6': p6, \
                         'p7':p7, 'p8':p8, 'p9':p9, 'p10': p10, 'p11':p11, 'pred_y': pred_y})
        csvfile.close()



