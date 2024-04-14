# # -*- coding: utf-8 -*-
# ## Stage 1: Installing dependencies and notebook gpu setup

import os, shutil
import json
os.environ['NCCL_P2P_DISABLE'] = "1"
# Commented out IPython magic to ensure Python compatibility.
#get a local copy of datasets
import sys
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
import datetime
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

def make_or_restore_model(cdir, checkpoints_exist=False):
  # Either restore the latest model, or create a fresh one
  # if there is no checkpoint available.
  if checkpoints_exist:
    checkpoints = [cdir + "/" + name for name in os.listdir(cdir)]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print("Restoring from", latest_checkpoint)
        return tf.keras.models.load_model(latest_checkpoint)
    print("Creating a new model")
  return get_compiled_model()

def train_model(cdir, model, X_train, y_train, epochs):
  # Directory for TensorBoard logs
  log_dir = tdir + "/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

  # Create a callback for model checkpoints
  checkpoint_cb = ModelCheckpoint(filepath=cdir + "/ckpt-{epoch}", save_freq="epoch")

  # Create a TensorBoard callback
  tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

  # Train the model
  model.fit(X_train, y_train, epochs=epochs, callbacks=[checkpoint_cb, tensorboard_callback])

def get_data_and_model(cdir, checkpoint_exists):
  X_train, y_train, X_test, y_test = get_dataset()

  model = make_or_restore_model(cdir, checkpoint_exists)
  return X_train, y_train, X_test, y_test, model

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
  print(nodes)
  # Define the cluster specification
  cluster_spec = {
      "worker": [f"{nodes[0]}:8000",f"{nodes[1]}:8001"]
  }

  # For the first worker (c22), set the task type and index in the TF_CONFIG environment variable to "worker" and 0, respectively:
  if argv[1] == "0":
    os.environ["TF_CONFIG"] = json.dumps({
        "cluster": cluster_spec,
        "task": {"type": "worker", "index": 0}  # This is for the first worker
    })

  # For the second worker (c23), set the task type and index in the TF_CONFIG environment variable to "worker" and 1, respectively:
  elif argv[1] == "1":
    os.environ["TF_CONFIG"] = json.dumps({
        "cluster": cluster_spec,
        "task": {"type": "worker", "index": 1}  # This is for the second worker
    })

  # For the evaluator, you would set the task type and index in the TF_CONFIG environment variable to "evaluator" and 0, respectively:
  elif argv[1] == "-1":
    os.environ["TF_CONFIG"] = json.dumps({
        "cluster": cluster_spec,
        "task": {"type": "evaluator", "index": 0}  # This is for the evaluator
    })
  # Check if the current task is evaluator
  user = os.environ.get("USER")
  if argv[1] == "-1":
    cdir = "/home/" + user + "/ckpt"
    tdir = "/home/" + user + "/tb"
  else:
    cdir = "/tmp/" + user + "/ckpt"
    tdir = "/tmp/" + user +"/tb"

  checkpoint_exists = True

  epochs = 15

  if (os.path.exists(cdir)):
    files = files = os.listdir(cdir)
    # if there is a file for every epoch of training this means training is done
    # and we need to remove them all
    if all(f'ckpt-{i}' in files for i in range(1, epochs + 1)):
      os.system(f"rm -rf {cdir}")
      os.system(f"rm -rf {tdir}")
      checkpoint_exists = False
  else:
    checkpoint_exists = False
  # we'll make sure these directories exist
  os.makedirs(cdir, exist_ok=True)
  os.makedirs(tdir, exist_ok=True)

  strategy = tf.distribute.MultiWorkerMirroredStrategy()

  with strategy.scope():
    X_train, y_train, X_test, y_test, model = get_data_and_model(cdir, checkpoint_exists)

  train_model(cdir, model, X_train, y_train, epochs)

    # Model evaluation and prediction
   # test_loss, test_accuracy = model.evaluate(X_test, y_test)

   # print("Test accuracy: {}".format(test_accuracy))

