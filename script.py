import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.metrics import r2_score

#load data from admissions_data.csv
dataset = pd.read_csv('admissions_data.csv')
labels = dataset.iloc[:,-1]
features =dataset.iloc[:,1:-1]

print(dataset.head())

print(dataset.describe())
print(features.head())

#data preprocessing

features_train, features_test, labels_train, labels_test =train_test_split(features, labels, 
                                                                           test_size = 0.2,random_state =40)
#scale numerical features
numerical_features=features.select_dtypes(include=['float64','int64'])
numerical_columns = numerical_features.columns
ct = ColumnTransformer([('scale', StandardScaler(),numerical_columns)], remainder='passthrough')

#fit the scale to the training data and convert from numpy arrays to pandas frame
features_train_scale =ct.fit_transform(features_train)

#applied the trained scale on the test data and convert from numpy arrays to pandas frame
features_test_scale=ct.transform(features_test)
#Create model
def creat_model():
    my_model = Sequential()
    num_features = features.shape[1]
    input = InputLayer(input_shape=(num_features,))
    my_model.add(input)
    my_model.add(Dense(8, activation='relu'))
    my_model.add(layers.Dropout(0.1))
    my_model.add(Dense(8, activation='relu'))
    my_model.add(layers.Dropout(0.2))
    my_model.add(Dense(1))
    print(my_model.summary())
    return my_model

#Initializing the optimizer and compiling model
my_model = creat_model()
opt = Adam(learning_rate=0.01)
my_model.compile(loss='mse',metrics=['mae'],optimizer=opt)

# apply early stopping for efficiency
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

# fit the model with 100 epochs and a batch size of 16
# validation split at 0.2
history = my_model.fit(features_train_scale, labels_train.to_numpy(), epochs=100, 
                       batch_size=16, verbose=1, validation_split=0.2, callbacks=[es])

# evaluate the model
val_mse, val_mae = my_model.evaluate(features_test_scale, labels_test.to_numpy(), verbose = 0)

# view the MAE performance
print("MAE: ", val_mae)
print("MSE: ", val_mse)
# evauate r-squared score
y_pred = my_model.predict(features_test_scale)

print('Score: ',r2_score(labels_test,y_pred))

# plot MAE and val_MAE over each epoch
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(history.history['mae'])
ax1.plot(history.history['val_mae'])
ax1.set_title('model mae')
ax1.set_ylabel('MAE')
ax1.set_xlabel('epoch')
ax1.legend(['train', 'validation'], loc='upper left')

# Plot loss and val_loss over each epoch
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_title('model loss')
ax2.set_ylabel('loss')
ax2.set_xlabel('epoch')
ax2.legend(['train', 'validation'], loc='upper left')

plt.show()