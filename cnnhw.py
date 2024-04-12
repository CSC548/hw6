# # -*- coding: utf-8 -*-
# ## Stage 1: Installing dependencies and notebook gpu setup
import os
os.environ['NCCL_P2P_DISABLE'] = "1"
# Commented out IPython magic to ensure Python compatibility.
#get a local copy of datasets
import sys
import tensorflow as tf
import json

from tensorflow.keras.datasets import cifar10

#for RTX GPUs
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# %matplotlib inline
tf.__version__

def get_compiled_model():
  model = tf.keras.models.Sequential()

  ### Adding the first CNN Layer

  #CNN layer hyper-parameters:
  #- filters: 32
  #- kernel_size:3
  #- padding: same
  #- activation: relu
  #- input_shape: (32, 32, 3)



  model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=[32, 32, 3]))

  ### Adding the second CNN Layer and max pool layer

  #CNN layer hyper-parameters:
  #- filters: 32
  #- kernel_size:3
  #- padding: same
  #- activation: relu

  #MaxPool layer hyper-parameters:
  #- pool_size: 2
  #- strides: 2
  #- padding: valid


  model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))

  model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

  ### Adding the third CNN Layer

  #CNN layer hyper-parameters:

  #    filters: 64
  #    kernel_size:3
  #    padding: same
  #    activation: relu
  #    input_shape: (32, 32, 3)



  model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))

  ###  Adding the fourth CNN Layer and max pool layer

  #CNN layer hyper-parameters:

  #    filters: 64
  #    kernel_size:3
  #    padding: same
  #    activation: relu

  #MaxPool layer hyper-parameters:

  #    pool_size: 2
  #    strides: 2
  #    padding: valid



  model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))

  model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

  ### Adding the Flatten layer

  model.add(tf.keras.layers.Flatten())

  ### Adding the first Dense layer

  #Dense layer hyper-parameters:
  #- units/neurons: 128
  #- activation: relu


  model.add(tf.keras.layers.Dense(units=128, activation='relu'))

  ### Adding the second Dense layer (output layer)

  #Dense layer hyper-parameters:

  # - units/neurons: 10 (number of classes)
  # - activation: softmax



  model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

  model.summary()

  ### Compiling the model

  #### sparse_categorical_accuracy
  #sparse_categorical_accuracy checks to see if the maximal true value is equal to the index of the maximal predicted value.

  #https://stackoverflow.com/questions/44477489/keras-difference-between-categorical-accuracy-and-sparse-categorical-accuracy


  model.compile(loss="sparse_categorical_crossentropy",
                optimizer="Adam", metrics=["sparse_categorical_accuracy"])
  return model

def get_dataset():
  #Setting class names for the dataset
  class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

  #Loading the dataset
  (X_train, y_train), (X_test, y_test) = cifar10.load_data()

  ### Image normalization

  X_train = X_train / 255.0

  X_train.shape

  X_test = X_test / 255.0

  return X_train, y_train, X_test, y_test




if __name__ == "__main__":
  argv = sys.argv

  if (len(argv) != 2):
    print("Invalid use of cnnhw you must only provide a job index")
    print("Proper usage: python cnnhw.py <number>")
    exit(1)
  # getting the list of nodes in the form ["c28", "c29"]
  nodes = os.environ.get('SLURM_NODELIST')
  nodes = nodes.replace("c", "")
  nodes = nodes.replace("[", "")
  nodes = nodes.replace("]", "")
  nodes = nodes.split("-")
  new_nodes = []
  for node in nodes:
    new_nodes.append("c" + node)
  nodes = new_nodes

  # reset the environment variable
  os.environ.pop('TF_CONFIG', None)
  cluster_spec = {
       "worker": [f"{nodes[0]}:2345", f"{nodes[1]}:1234"],
       "evaluator": [f"{nodes[0]}:1235"]
   }

  cluster_spec = tf.train.ClusterSpec(cluster_spec)
  resolver = None
  # For the first worker (c22), set the task type and index in the TF_CONFIG environment variable to "worker" and 0, respectively:
  if argv[1] == "0":
    os.environ["TF_CONFIG"] = json.dumps({
        "cluster": cluster_spec,
        "task": {"type": "worker", "index": 0}  # This is for the first worker
    })
    resolver = SimpleClusterResolver(cluster_spec, task_type="worker",
                                        task_id=0)

  # For the second worker (c23), set the task type and index in the TF_CONFIG environment variable to "worker" and 1, respectively:
  elif argv[1] == "1":
    os.environ["TF_CONFIG"] = json.dumps({
        "cluster": cluster_spec,
        "task": {"type": "worker", "index": 1}  # This is for the second worker
    })
    resolver = SimpleClusterResolver(cluster_spec, task_type="worker",
                                        task_id=0)

  # For the evaluator, you would set the task type and index in the TF_CONFIG environment variable to "evaluator" and 0, respectively:
  elif argv[1] == "-1":
    os.environ["TF_CONFIG"] = json.dumps({
        "cluster": cluster_spec,
        "task": {"type": "evaluator", "index": 0}  # This is for the evaluator
    })
    resolver = SimpleClusterResolver(cluster_spec, task_type="evaluator",
                                        task_id=1)

  print(cluster_spec)
  X_train, y_train, X_test, y_test = get_dataset()

  # Training the model


  strategy = tf.distribute.MultiWorkerMirroredStrategy(cluster_spec)
  with strategy.scope():
    model = get_compiled_model()
    model.fit(X_train, y_train, epochs=15)

  # Model evaluation and prediction


  # print("Test accuracy: {}".format(test_accuracy))

