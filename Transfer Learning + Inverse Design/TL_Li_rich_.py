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
import csv
import random
import random as rd
import os
import tensorflow as tf
from datetime import datetime
rand_state = 42

os.environ['PYTHONHASHSEED']=str(rand_state)
tf.random.set_seed(rand_state)
np.random.seed(rand_state)
random.seed(rand_state)

step_parameter = 10
Match_found = 25
req_Y = 370
start = datetime.now()
#req_Y = " ".join(map(str, req_Y)).replace('[', ' ')
#req_Y = [x for x in req_Y.split()]
#req_Y = float(req_Y)
print(req_Y)
lb = [1,    0,     0, 0,  550,  8,  20,  4,  0.05]
ub = [1.7,  0.6, 0.4, 0,  1300, 25, 60, 5.1, 1.1]

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

nSNP = X_scaled_train.shape[1]
model = load_model('NCM_model.h5')
model_TL = Sequential()
model_TL.add(Conv1D(256, kernel_size=3, input_shape=(nSNP, 1), activation='tanh'))
for i in range(4):
    i = i+1
    layer = model.layers[i]
    layer.trainable = False
    model_TL.add(layer)
model_TL.add(Dropout(0.3))
#model_TL.add(Dense(32, activation='relu', name='TL_layer0'))
model_TL.add(Dense(16, activation='relu', name='TL_layer1'))
model_TL.add(Dense(Y_scaled_train.shape[1], name='TL_layer2'))
model_TL.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
path = "./Li_model.h5"
model_TL.summary()
checkpoint = ModelCheckpoint(path, monitor='val_loss', save_best_only=True, mode='min', verbose=0)

# fit model
history = model_TL.fit(X_scaled_train, Y_scaled_train, validation_data=(X_scaled_test, Y_scaled_test), epochs=500, verbose=0,  callbacks=[checkpoint])

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
#plt.show()

############## Experiment validation ######################
exp_data = pd.read_csv('./Experimental_data.csv')
exp_fea = ['Li', 'Ni ', 'Co', 'Mn',  'sin_temp', 'sin_t', 'm_temp', 'cut_off', 'c_rate']
X_exp = exp_data.loc[:,exp_fea].values #data_import[features].values

Exp_scaledX = scaler.transform(X_exp)
Exp_scaledX = np.expand_dims(Exp_scaledX, axis=2)
Exp_pred_Y = model_TL.predict(Exp_scaledX)
Exp_pred_Y = scaler2.inverse_transform(Exp_pred_Y)
#Exp_pred_Y = Exp_pred_Y.squeeze()

print(Exp_pred_Y)
with open("Predict_experiment_Y.csv", "w", newline='') as file_exp:
    np.savetxt(file_exp, Exp_pred_Y, fmt='%d ')


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
with open("RESULT.csv", "w", newline='') as file3:
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

MF = 0
#new_param_sol = []
#match_prediction = []
#iter = 0
while MF < Match_found:
    #iter = iter + 1
    for i in range(step_parameter):
        p1 = round(rd.uniform(lb[0], ub[0]), 3)
        p2 = round(rd.uniform(lb[1], ub[1]), 3)
        p3 = round(rd.uniform(lb[2], ub[2]), 3)
        p4 = round((2 - p1 - p2 - p3), 3)
        p5 = round(rd.uniform(lb[4], ub[4]), 2)
        p6 = round(rd.uniform(lb[5], ub[5]), 2)
        p7 = round(rd.uniform(lb[6], ub[6]), 2)
        p8 = round(rd.uniform(lb[7], ub[7]), 2)
        p9 = round(rd.uniform(lb[8], ub[8]), 2)
        if p4 >= 0:
            new_param_ = np.hstack((p1, p2, p3, p4, p5, p6, p7, p8, p9)).reshape(1, -1)
            #print(new_param_)
            new_param_Scale = scaler.transform (new_param_)
            new_param_Scale = np.expand_dims(new_param_Scale, axis=2)
            y_predict = model_TL.predict(new_param_Scale)
            pred_y = scaler2.inverse_transform(y_predict).round()
            #print(pred_y)
            #pred_y = " ".join(map(str, y_predict)).replace('[', ' ').replace(']', ' ').replace("'", ' ').replace(".", ' ')
            #pred_y = [xi for xi in pred_y.split()]

            #print("................Finding Solutions: ", iter, ", Be patient............")
            #with open('result_.csv', 'a', newline='') as csvfile:
            #    fieldnames = ['pred_y']
            #    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
            #    writer.writerow({'pred_y': pred_y})
            #    csvfile.close()
            if (req_Y*(1-0.05)) <= pred_y <= (req_Y*(1+0.05)):
                MF += 1
                print('Found', MF,'Solution')
                with open('RESULT_TL_Li_prediction_370_new_.csv', 'a', newline='') as csvfile:
                    fieldnames = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'pred_y']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
                    writer.writerow({'p1': p1, 'p2': p2, 'p3': p3, 'p4': p4, 'p5': p5, 'p6': p6, \
                                     'p7':p7, 'p8':p8, 'p9':p9, 'pred_y': pred_y})
                    csvfile.close()

end = datetime.now()
time = end-start
print('Calculation time (hh:mm:ss.ms): {}'.format(time))
with open('./calculation_time.dat', 'a') as result4:
    result4.writelines("Calculation time (hh:mm:ss.ms): %s\n" % format(time))