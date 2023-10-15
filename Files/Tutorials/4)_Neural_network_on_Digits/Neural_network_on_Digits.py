# Source: https://medium.com/swlh/data-annotation-using-active-learning-with-python-code-aa5b1fe13608

import random
import os
import numpy as np
import pandas as pd
import scipy as sp
import sklearn
from sklearn import datasets
from sklearn.metrics import accuracy_score
import tensorflow as tf

digits = datasets.load_digits()

def get_initial_labelled_dataset(num_samples_per_class=5):
  """
  Getting the initial balanced dataset. Although this is random in this code.
  In actual production system it should be informative labelled dataset given by the annotators/SMEs.

  Args:
    num_samples_per_class: int, number of samples to be takes per class in the first iteration.

  Returns:
    numpy array of labelled dataset with true labels and pooled dataset with true labels (for performance checking)
  """

  X = list()
  y = list()
  X_pooled = list()
  y_pooled = list()
  labelled_idx = list()

  counter_dict = dict()

  for idx, target in enumerate(digits['target']):
    if target in counter_dict:
      if counter_dict[target] == num_samples_per_class:
        continue
      counter_dict[target] += 1
    else:
      counter_dict[target] = 1
    X.append(digits['data'][idx])
    y.append(target)
    labelled_idx.append(idx)

  X_pooled = np.delete(digits['data'], labelled_idx, axis=0)
  y_pooled = np.delete(digits['target'], labelled_idx)

  return np.asarray(X), np.asarray(y), X_pooled, y_pooled

X, y, X_pooled, y_pooled = get_initial_labelled_dataset()

print(X.shape)
print(y.shape)
print(X_pooled.shape)
print(y_pooled.shape)

# Create Classification Model - Keras Model

def create_model():
  # Input Layer
  input_layer = tf.keras.Input(shape=(64,), name='input_layer') # Feature dimension=64
  input_dropout_layer = tf.keras.layers.Dropout(0.4, name='input_dropout_layer')(input_layer, training=True) #training=True activates MC Dropout

  # Hidden Layer 1
  dense_layer_1 = tf.keras.layers.Dense(256, activation='relu', name='dense_layer_1')(input_dropout_layer)
  dropout_layer_1 = tf.keras.layers.Dropout(0.4, name='dropout_layer_1')(dense_layer_1, training=True)

  # Hidden Layer 2
  dense_layer_2 = tf.keras.layers.Dense(256, activation='relu', name='dense_layer_2')(dropout_layer_1)
  dropout_layer_2 = tf.keras.layers.Dropout(0.4, name='dropout_layer_2')(dense_layer_2, training=True)

  # Hidden Layer 3
  dense_layer_3 = tf.keras.layers.Dense(256, activation='relu', name='dense_layer_3')(dropout_layer_2)
  dropout_layer_3 = tf.keras.layers.Dropout(0.4, name='dropout_layer_3')(dense_layer_3, training=True)

  # Output Layer
  output_layer = tf.keras.layers.Dense(len(digits['target_names']), activation='softmax', name='output_layer')(dropout_layer_3)

  # Model Init
  model = tf.keras.Model(inputs=input_layer, outputs=output_layer, name='Model')

  return model

def max_entropy_acquisition(X_pool, T, num_query):
  """
  Calculate the entropy of the ensempe of models (by using MC Dropout).
  Get the max entropy and return the index of the data

  Args:
    X_pool: ndarray, Pooled dataset (unlabelled)
    T: int, number of iteration to replicate ensemble model using MC Dropout
    num_query: int, number of datapoints to be returned by the model per itration

  Returns:
    Index points of uncertain dataset
    mean entropy value
  """

  proba_all = np.zeros(shape=(X_pool.shape[0], 10))
  for _ in range(T):
    probas = model.predict(X_pool)
    proba_all += probas
  avg_proba = np.divide(proba_all, T)
  entropy_avg = sp.stats.entropy(avg_proba, base=10, axis=1)
  uncertain_idx = entropy_avg.argsort()[-num_query:][::-1]
  return uncertain_idx, entropy_avg.mean()

def manage_data(X, y, X_pooled, y_pooled, idx):
  """
  After every iteration, the uncertain unlabelled data will be given the true labels (in reality the manual annotator will label the data)

  Args:
    X: ndarray, Training Features
    y: ndarray, Training Labels (true)
    X_pooled: ndarray, Unlabelled dataset
    y_pooled: ndarray, True labels of the unlabelled dataset
    idx: ndarray,list, uncertain data index

  Returns:
    Lablled and unlabelled dataset
  """

  pool_mask = np.ones(len(X_pooled), dtype=bool)
  pool_mask[idx] = False
  pool_mask_2 = np.zeros(len(X_pooled), dtype=bool)
  pool_mask_2[idx] = True
  new_training = X_pooled[pool_mask_2]
  new_label = y_pooled[pool_mask_2]
  X_pooled = X_pooled[pool_mask]
  y_pooled = y_pooled[pool_mask]
  X = np.concatenate([X, new_training])
  y = np.concatenate([y, new_label])

  return X, y, X_pooled, y_pooled

# Main loop
for i in range(30):
  print('*'*50)
  model = create_model()
  custom_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
  model.compile(optimizer='adam', loss=custom_loss, metrics=['accuracy'])
  model.fit(X, y, epochs=20, batch_size=8, verbose=0)
  uncertain_idx, entropy_avg = max_entropy_acquisition(X_pooled, 100, 20)
  print('Average Entropy: {}'.format(entropy_avg))
  X, y, X_pooled, y_pooled = manage_data(X, y, X_pooled, y_pooled, uncertain_idx)
  print('Iteration Done: {}'.format(i+1))