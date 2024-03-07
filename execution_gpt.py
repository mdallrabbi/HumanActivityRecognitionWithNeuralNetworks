import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras_tuner.tuners import RandomSearch
from tensorflow import keras
from tensorflow.keras import layers
from keras_tuner.tuners import RandomSearch
import warnings

# Ignore warnings
warnings.filterwarnings('ignore')

# Reading The Data
train_data = pd.read_csv('/home/rabbi/research/HumanActivityRecognitionWithNeuralNetworks/train.csv')
test_data = pd.read_csv('/home/rabbi/research/HumanActivityRecognitionWithNeuralNetworks/test.csv')

# Data Analysis
print(f'Shape of train data is: {train_data.shape}\nShape of test data is: {test_data.shape}')

# Displaying data
pd.set_option("display.max_columns", None)
print(train_data.head())
print(train_data.columns)
print(train_data.describe())
print(train_data['Activity'].unique())
train_data['Activity'].value_counts().sort_values().plot(kind='bar', color='pink')
plt.show()

# Data preprocessing
x_train, y_train = train_data.iloc[:, :-2], train_data.iloc[:, -1:]
x_test, y_test = test_data.iloc[:, :-2], test_data.iloc[:, -1:]
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)
scaling_data = MinMaxScaler()
x_train = scaling_data.fit_transform(x_train)
x_test = scaling_data.transform(x_test)

# Building the initial model
model = Sequential()
model.add(Dense(units=64, kernel_initializer='normal', activation='sigmoid', input_dim=x_train.shape[1]))
model.add(Dropout(0.2))
model.add(Dense(units=6, kernel_initializer='normal', activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training the initial model
history = model.fit(x_train, y_train, batch_size=64, epochs=10, validation_data=(x_test, y_test))

# Hyperparameter Tuning
def build_model(hp):
    model = keras.Sequential()
    for i in range(hp.Int('num_layers', 2, 25)):
        model.add(layers.Dense(units=hp.Int('units' + str(i), min_value=32, max_value=512, step=32),
                               kernel_initializer=hp.Choice('initializer', ['uniform', 'normal']),
                               activation=hp.Choice('activation', ['relu', 'sigmoid', 'tanh'])))
    model.add(layers.Dense(6, kernel_initializer=hp.Choice('initializer', ['uniform', 'normal']), activation='softmax'))
    model.add(Dropout(0.2))
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    return model

tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=3,
    directory='project', project_name='Human_activity_recognition')

tuner.search(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Getting the best model
best_model = tuner.get_best_models(num_models=1)[0]

# Training the best model
history = best_model.fit(x_train, y_train, epochs=51, validation_data=(x_test, y_test))

# Plotting the training history
accuracy = history.history['accuracy']
loss = history.history['loss']
validation_loss = history.history['val_loss']
validation_accuracy = history.history['val_accuracy']

plt.figure(figsize=(15, 7))

# Accuracy plot
plt.subplot(2, 2, 1)
plt.plot(range(51), accuracy, label='Training Accuracy')
plt.plot(range(51), validation_accuracy, label='Validation Accuracy')
plt.legend(loc='upper left')
plt.title('Accuracy : Training Vs Validation')

# Loss plot
plt.subplot(2, 2, 2)
plt.plot(range(51), loss, label='Training Loss')
plt.plot(range(51), validation_loss, label='Validation Loss')
plt.title('Loss : Training Vs Validation ')
plt.legend(loc='upper right')
plt.show()

# Displaying the best model
print(best_model.summary())