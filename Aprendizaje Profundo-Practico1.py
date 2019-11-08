#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 21:29:50 2019

@author: simon
"""

"""Exercise 1

Usage:

$ CUDA_VISIBLE_DEVICES=2 python practico_1_train_petfinder.py --dataset_dir ../ --epochs 30 --dropout 0.1 0.1 --hidden_layer_sizes 200 100

To know which GPU to use, you can check it with the command

$ nvidia-smi
"""

import argparse

import os
import mlflow
import numpy
import pandas
import tensorflow as tf
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Embedding, Flatten, concatenate, Concatenate, Input
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow import keras
from tensorflow.keras.models import Model


TARGET_COL = 'AdoptionSpeed'
tf.__version__

def read_args():
    parser = argparse.ArgumentParser(
        description='Training a MLP on the petfinder dataset')
    # Here you have some examples of classifier parameters. You can add
    # more arguments or change these if you need to.
    parser.add_argument('--dataset_dir', default='./petfinder_dataset', type=str,
                        help='Directory with the training and test files.')
    parser.add_argument('--hidden_layer_sizes', nargs='+', default=[256], type=int,
                        help='Number of hidden units of each hidden layer.')
    parser.add_argument('--epochs', default=100, type=int,
                        help='Number of epochs to train.')
    parser.add_argument('--dropout', nargs='+', default=[0.3], type=float,
                        help='Dropout ratio for every layer.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Number of instances in each batch.')
    parser.add_argument('--experiment_name', type=str, default='Modelo 4',
                        help='Name of the experiment, used in mlflow.')
    args = parser.parse_args()

    assert len(args.hidden_layer_sizes) == len(args.dropout)
    return args


def process_features(df, one_hot_columns, numeric_columns, embedded_columns, test=False):
    direct_features = []

    # Create one hot encodings
    for one_hot_col, max_value in one_hot_columns.items():
        direct_features.append(tf.keras.utils.to_categorical(df[one_hot_col] - 1, max_value))

    for numeric_column in numeric_columns:
        
    
    # TODO Create and append numeric columns
    # Don't forget to normalize!
    # ....
    
    # Concatenate all features that don't need further embedding into a single matrix.
    features = {'direct_features': numpy.hstack(direct_features)}

    # Create embedding columns - nothing to do here. We will use the zero embedding for OOV
    for embedded_col in embedded_columns.keys():
        features[embedded_col] = df[embedded_col].values

    if not test:
        nlabels = df[TARGET_COL].unique().shape[0]
        # Convert labels to one-hot encodings
        targets = tf.keras.utils.to_categorical(df[TARGET_COL], nlabels)
    else:
        targets = None
    
    return features, targets



def load_dataset(dataset_dir, batch_size):

    # Read train dataset (and maybe dev, if you need to...)
    dataset, dev_dataset = train_test_split(
        pandas.read_csv(os.path.join(dataset_dir, 'train.csv')), test_size=0.2)
    
    test_dataset = pandas.read_csv(os.path.join(dataset_dir, 'test.csv'))
    
    print('Training samples {}, test_samples {}'.format(
        dataset.shape[0], test_dataset.shape[0]))
    
    return dataset, dev_dataset, test_dataset


def main():
    args = read_args()
    dataset, dev_dataset, test_dataset = load_dataset(args.dataset_dir, args.batch_size)
    nlabels = dataset[TARGET_COL].unique().shape[0]
    
    # It's important to always use the same one-hot length
    one_hot_columns = {
        one_hot_col: dataset[one_hot_col].max()
        for one_hot_col in ['Gender', 'Color1']
    }
    embedded_columns = {
        embedded_col: dataset[embedded_col].max() + 1
        for embedded_col in ['Breed1']
    }
    numeric_columns = ['Age', 'Fee']
    
    # TODO (optional) put these three types of columns in the same dictionary with "column types"
    X_train, y_train = process_features(dataset, one_hot_columns, numeric_columns, embedded_columns)
    direct_features_input_shape = (X_train['direct_features'].shape[1],)
    X_dev, y_dev = process_features(dev_dataset, one_hot_columns, numeric_columns, embedded_columns)
    
    # Create the tensorflow Dataset
    batch_size = 32
    # TODO shuffle the train dataset!
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
    dev_ds = tf.data.Dataset.from_tensor_slices((X_dev, y_dev)).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices(process_features(
        test_dataset, one_hot_columns, numeric_columns, embedded_columns, test=True)[0]).batch(batch_size)

##########################################################################

# Modelo Base

modelo="Modelo Base"
epochs=100
hidden_n=256
batch_size=32

dataset, dev_dataset, test_dataset = load_dataset(args.dataset_dir, args.batch_size)


dataset[["Gender","Color1","Breed1"]] = dataset[["Gender","Color1","Breed1"]].astype('category')
dev_dataset[["Gender","Color1","Breed1"]] = dataset[["Gender","Color1","Breed1"]].astype('category')
test_dataset[["Gender","Color1","Breed1"]] = dataset[["Gender","Color1","Breed1"]].astype('category')


# Create a minimum and maximum processor object
min_max_scaler = preprocessing.MinMaxScaler()

# Create an object to transform the data to fit minmax processor
x_scaled = min_max_scaler.fit_transform(dataset[["Age","Fee"]])
dataset[["Age","Fee"]]=x_scaled
dataset1= pandas.get_dummies(dataset[["Gender","Age","Fee"]])


# Mismo para dev
x_scaled_dev = min_max_scaler.fit_transform(dev_dataset[["Age","Fee"]])
dev_dataset[["Age","Fee"]]=x_scaled_dev
dev_dataset1= pandas.get_dummies(dev_dataset[["Gender","Age","Fee"]])
#dev_dataset1= pandas.get_dummies(dev_dataset[["Gender","Color1","Age","Fee","MaturitySize", 'Breed1']])

xtra=dataset1
ytra=dataset['AdoptionSpeed']
ytra = ytra.astype("category")
ytra= pandas.get_dummies(ytra)
X_dev=dev_dataset1
y_dev=dev_dataset['AdoptionSpeed']
y_dev = y_dev.astype("category")
y_dev= pandas.get_dummies(y_dev)

# Mismo para test
x_scaled_test = min_max_scaler.fit_transform(test_dataset[["Age","Fee"]])
test_dataset[["Age","Fee"]]=x_scaled_test
test_dataset1= pandas.get_dummies(test_dataset[["Gender","Age","Fee"]])
x_test=test_dataset1



#get number of columns in training data
n_cols = 5

# Modelo Base

model = Sequential()
model.add(Dense(hidden_n,activation="relu",input_shape=(n_cols,)))
model.add(Dense(5,activation="linear"))
model.compile(optimizer="adam", loss='mean_squared_error',metrics=["acc"])
model.summary()    # TODO: Fit the model
# Train
history = model.fit(xtra,ytra, epochs=epochs,batch_size=batch_size)
performance = model.evaluate(X_dev, y_dev)


##########################################################################

#################            2


# Modelo 2

modelo="Modelo 2"
epochs=100
hidden_n=256
batch_size=32

# Modelo 2

model = Sequential()
model.add(Dense(hidden_n,activation="relu",input_shape=(n_cols,)))
model.add(Dense(5,activation="softmax"))
model.compile(optimizer="adam", loss='categorical_crossentropy',metrics=["acc"])
model.summary()    # TODO: Fit the model
# Train
history = model.fit(xtra,ytra, epochs=epochs,batch_size=batch_size)
performance = model.evaluate(X_dev, y_dev)



##########################################################################

#################            3


# Modelo 3

modelo="Modelo 3"
epochs=100
hidden_n=256
dropout=0.3
batch_size=32

# Modelo 3

#create model
model = Sequential()
model.add(Dense(hidden_n,activation="relu",input_shape=(n_cols,),kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(Dropout(args.dropout[0]))
model.add(Dense(hidden_n/2,activation="relu",input_shape=(n_cols,),kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(Dropout(args.dropout[0]))
model.add(Dense(5,activation="softmax"))

model.compile(optimizer="adam", loss='categorical_crossentropy',metrics=["acc"])
model.summary()
history = model.fit(xtra,ytra, epochs=epochs,batch_size=batch_size)
performance = model.evaluate(X_dev, y_dev)


##########################################################################

# Modelo 4 

modelo="Modelo 4"
epochs=100
hidden_n=256
dropout=0.5
batch_size=32

dataset, dev_dataset, test_dataset = load_dataset(args.dataset_dir, args.batch_size)
dataset[["Gender","Color1","Breed1"]] = dataset[["Gender","Color1","Breed1"]].astype('category')
dev_dataset[["Gender","Color1","Breed1"]] = dataset[["Gender","Color1","Breed1"]].astype('category')
test_dataset[["Gender","Color1","Breed1"]] = dataset[["Gender","Color1","Breed1"]].astype('category')


# Create a minimum and maximum processor object
min_max_scaler = preprocessing.MinMaxScaler()
# Create an object to transform the data to fit minmax processor
x_scaled = min_max_scaler.fit_transform(dataset[["Age","Fee","MaturitySize","Health","Quantity"]])
dataset[["Age","Fee","MaturitySize","Health","Quantity"]]=x_scaled
dataset1= pandas.get_dummies(dataset[["Gender","Age","Fee","MaturitySize","Health","Quantity"]])

# Mismo para dev
x_scaled_dev = min_max_scaler.fit_transform(dev_dataset[["Age","Fee","MaturitySize","Health","Quantity"]])
dev_dataset[["Age","Fee","MaturitySize","Health","Quantity"]]=x_scaled_dev
dev_dataset1= pandas.get_dummies(dev_dataset[["Gender","Age","Fee","MaturitySize","Health","Quantity"]])

xtra=dataset1
ytra=dataset['AdoptionSpeed']
ytra = ytra.astype("category")
ytra= pandas.get_dummies(ytra)
X_dev=dev_dataset1
y_tdev=dev_dataset['AdoptionSpeed']
y_dev = y_test.astype("category")
y_dev= pandas.get_dummies(y_dev)


# Mismo para test
x_scaled_test = min_max_scaler.fit_transform(test_dataset[["Age","Fee","MaturitySize","Health","Quantity"]])
test_dataset[["Age","Fee","MaturitySize","Health","Quantity"]]=x_scaled_test
test_dataset1= pandas.get_dummies(test_dataset[["Gender","Age","Fee","MaturitySize","Health","Quantity"]])
predict=test_dataset1


#get number of columns in training data
n_cols = 8
#create model
model = Sequential()
model.add(Dense(hidden_n,activation="relu",input_shape=(n_cols,),kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(Dropout(args.dropout[0]))
model.add(Dense(hidden_n/2,activation="relu",input_shape=(n_cols,),kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(Dropout(args.dropout[0]))
model.add(Dense(5,activation="softmax"))
model.compile(optimizer="adam", loss='categorical_crossentropy',metrics=["acc"])
model.summary()    # TODO: Fit the model
history = model.fit(tra,yb, epochs=epochs,batch_size=batch_size)
performance = model.evaluate(X_test, y_test)


###############################################################################

# Modelo 5

dataset, dev_dataset, test_dataset = load_dataset(args.dataset_dir, args.batch_size)
dataset[["Gender","Color1"]] = dataset[["Gender","Color1"]].astype('category')
dev_dataset[["Gender","Color1"]] = dataset[["Gender","Color1"]].astype('category')
test_dataset[["Gender","Color1"]] = dataset[["Gender","Color1"]].astype('category')
dataset[["Gender","Color1","Breed1"]] = dataset[["Gender","Color1","Breed1"]].astype('category')
dev_dataset[["Gender","Color1","Breed1"]] = dataset[["Gender","Color1","Breed1"]].astype('category')
test_dataset[["Gender","Color1","Breed1"]] = dataset[["Gender","Color1","Breed1"]].astype('category')


# Create a minimum and maximum processor object
min_max_scaler = preprocessing.MinMaxScaler()
# Create an object to transform the data to fit minmax processor
x_scaled = min_max_scaler.fit_transform(dataset[["Age","Fee","MaturitySize","Health","Quantity"]])
dataset[["Age","Fee","MaturitySize","Health","Quantity"]]=x_scaled
dataset1= pandas.get_dummies(dataset[["Gender","Age","Fee","MaturitySize","Health","Quantity"]])
emb=pandas.get_dummies(dataset[["Breed1"]])


# Add one input and one embedding for each embedded column


embedding_size = int(max_value / 4)
max_value = max(dataset["Breed1"])


first_input = Input(shape=(1, ))
first_dense = Embedding(input_dim=max_value+1,output_dim=embedding_size)(first_input)
first=Flatten()(first_dense)

second_input = Input(shape=(n_cols, ))
second_dense = Dense(256, activation="relu",kernel_regularizer=l2(0.01))(second_input)
second_batch= BatchNormalization()(second_dense)
second_drop=Dropout(args.dropout[0])(second_batch)

merge_one = concatenate([first, second_dense])
merge_1 =Dense(128,activation="relu",kernel_regularizer=l2(0.01))(merge_one)
merge_batch= BatchNormalization()(merge_1)
merge_drop=Dropout(args.dropout[0])(merge_batch)
merge = Dense(5,activation="softmax")(merge_drop)
model = Model(inputs=[first_input, second_input], outputs=merge)
model.compile(optimizer="adam", loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
history = model.fit([dataset["Breed1"],xtra],ytra, epochs=epochs,batch_size=batch_size)

performance = model.evaluate([dev_dataset["Breed1"].fillna(0),X_dev, y_dev)


# Predictions for the Kaggle competition.

predictions = model.predict([test_dataset["Breed1"].fillna(0),predict])
predicted = numpy.argmax(predictions, axis=1)
predictions=predicted
predictions=numpy.array(predictions).ravel()    
predictions=predictions.astype(numpy.int)    
    
submission = pandas.DataFrame(list(zip(test_dataset["PID"], numpy.array(predictions).ravel())), columns=["PID", "AdoptionSpeed"])
submission.to_csv("submission.csv", header=True, index=False)
