import os
import random, shutil
import logging
from functools import partial
from collections import namedtuple
import argparse
import string

import aug_lib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
import seaborn as sns

import keras
import tensorflow as tf
from tensorflow.keras.preprocessing import image

from tensorflow.keras.applications.efficientnet import EfficientNetB7
from tensorflow.keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications import ResNet50

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2D, MaxPool2D, Activation, Dropout, BatchNormalization

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import SGD

# setting up GPU configuration
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.compat.v1.Session(config=config)

keras.backend.set_session(session)

from tensorflow.python.client import device_lib
local_devices_str = str(device_lib.list_local_devices())
if "GPU" not in local_devices_str and "TPU" not in local_devices_str:
    logging.info("Working with CPU")

# definition of arguments for starting the program
parser = argparse.ArgumentParser(description="Tensorflow LungColonCancer Training")
parser.add_argument("--epochs", default=100, type=int, help="set epochs, default=100")
parser.add_argument("--trivialaug", action="store_true", help="use TrivialAugment")
parser.add_argument("--randaug", action="store_true", help="use RandAugment")
parser.add_argument("--tag", action="store", default="notag", dest="tag", help="use setting tag e.g. multiple runs")


args = parser.parse_args()

epochs = args.epochs
tag = args.tag

# set path of data directory with just diretories of classes in it and no separation of lung and colon directories   
# like that:
# ../LungColon
#   |- colon_aca
#       |- colonca2.jpeg
#   |- colon_n
#   |- lung_aca
#   |- lung_n
#   |- lung_scc
data_dir = "/projects/p_al4ml/datasets/LungColon"

# choosing augmentation strategy based on parameter input
if args.trivialaug:
    print("TrivialAugment will be used!")
    def trivial_process(image):
        image = Image.fromarray(image.astype('uint8'),'RGB')
        augment = aug_lib.TrivialAugment()
        image = augment(image)
        image = np.array(image)
        return image
    preprocessing_function = trivial_process
    
elif args.randaug:
    print("RandAugment will be used!")
    def rand_process(image):
        image = Image.fromarray(image.astype('uint8'),'RGB')
        augment = aug_lib.RandAugment(2, 30)
        image = augment(image)
        image = np.array(image)
        return image
    preprocessing_function = rand_process

else:
    print("NoAugment will be used!")
    preprocessing_function = None

# creating config of data generators, only train data will be augmented for comparison with results of original data
train_datagen = image.ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True,preprocessing_function=preprocessing_function)
valid_datagen = image.ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
test_datagen = image.ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)

BATCH_SIZE = 15 # Change batch size, if you run out of memory! Or increase it, if you have a lot of RAM (>> 16 Gb) for faster trainig
IMG_SHAPE = (224, 224, 3)
CLASSES_NUM = 5
ACCURACY_THRESHOLD = 0.99

# using dataframes for division of dataset in train, valid and test set without moving of files into different directories 
traindf = pd.DataFrame(columns=['image', 'label'])
valdf = pd.DataFrame(columns=['image', 'label'])
testdf = pd.DataFrame(columns=['image', 'label'])

sourcedir = [f"{data_dir}/lung_scc",f"{data_dir}/lung_n",f"{data_dir}/lung_aca",f"{data_dir}/colon_aca",f"{data_dir}/colon_n"]

# divide dataset randomly in train, valid and test set with a distribution of 80%-10%-10%
for i in range(5):
    dir = os.listdir(path=sourcedir[i])
    category = sourcedir[i].split('/')[-1]
    random.shuffle(dir)
    amount = len(dir)
    for j in range(int(amount*0.8)):
        file= category + '/' + dir[j]       
        traindf = pd.concat([traindf,pd.DataFrame([[file, category]],columns=['image','label'])])
    for k in range(int(amount*0.8), int(amount*0.9)):
        file= category + '/' + dir[k]
        valdf = pd.concat([valdf,pd.DataFrame([[file,category]],columns=['image','label'])])
    for l in range(int(amount*0.9), amount):
        file=  category + '/' + dir[l]
        testdf = pd.concat([testdf,pd.DataFrame([[file,category]],columns=['image','label'])])

# creating train, valid and test data generators
print('Trainig data:\n')
train_generator=train_datagen.flow_from_dataframe(
    dataframe=traindf,
    directory=data_dir + '/',
    x_col="image",
    y_col="label",
    batch_size=BATCH_SIZE,
    seed=666,
    shuffle=True,
    class_mode="categorical",
    target_size=IMG_SHAPE[:2],
    validate_filenames=False)

print('Validating data:\n')
valid_generator=valid_datagen.flow_from_dataframe(
    dataframe=valdf,
    directory=data_dir + '/',
    x_col="image",
    y_col="label",
    batch_size=BATCH_SIZE,
    seed=666,
    shuffle=True,
    class_mode="categorical",
    target_size=IMG_SHAPE[:2],
    validate_filenames=False)

# test data generator without shuffle and label
print('Test data:\n')
test_generator=test_datagen.flow_from_dataframe(
    dataframe=testdf,
    directory=data_dir + '/',
    x_col="image",
    y_col=None,
    batch_size=1,
    seed=666,
    shuffle=False,
    class_mode=None,
    target_size=IMG_SHAPE[:2],
    validate_filenames=False)


STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size


# Set to True to train simple from scratch CNN
RUN_SIMPLE_CNN = True
# Set to True to use fine-tuning
RUN_PRETRAINED = False
# Set to True to use an existing but not pretrained model
RUN_EXISTING = False

RUN_BASIC_CNN = False

def top_layer(output):
    layer = GlobalAveragePooling2D()(output)
    layer = Flatten()(layer)
    layer = Dense(128, activation='relu')(layer)
    # layer = Dropout(0.5)(layer)
    # Drop out layer is optional: you may include it to fix overfitting
    layer = Dense(64, activation='relu')(layer)
    layer = Dense(CLASSES_NUM, activation='softmax')(layer)
    return layer


if RUN_SIMPLE_CNN:

    # Features extraction layers
    model = tf.keras.models.Sequential([
    
    Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=IMG_SHAPE),
    BatchNormalization(),
    MaxPool2D(2),
    
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPool2D(2),
  
    Conv2D(128, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPool2D(2)
    ])

    # Classification layers
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(CLASSES_NUM, activation='softmax'))

    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',metrics=['accuracy'])

    name = f"SimpleCNN_{tag}" 

    checkpointing = ModelCheckpoint(
        name,
        save_weights_only=True,
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    )

    log_saving = CSVLogger(f'{name}-log.csv', append=True, separator=';')


    print(f'{name} IS READY FOR TRAINING:')
    model.summary()
    
    print(f'STARTED TRAINIG OF {name} FROM SCRATCH MODEL')

    model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=epochs,
        shuffle=True,
        callbacks=[checkpointing, log_saving]
    )

    print(f'TRAINIG OF {name} COMPLETED')
    print('------------------------------')

if RUN_BASIC_CNN:
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=IMG_SHAPE))
    model.add(Activation("relu"))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.6))

    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.6))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(CLASSES_NUM))
    model.add(Activation("softmax"))

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001, decay=1e-6)
    # optimizer = keras.optimizers.Adadelta(learning_rate=1.0, rho=0.95, epsilon=None, decay=0.0)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )

    name = f"BasicCNN_{tag}" 

    checkpointing = ModelCheckpoint(
        name,
        save_weights_only=True,
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    )

    log_saving = CSVLogger(f'{name}-log.csv', append=True, separator=';')


    print(f'{name} IS READY FOR TRAINING:')
    model.summary()
    
    print(f'STARTED TRAINIG OF {name} FROM SCRATCH MODEL')

    model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=epochs,
        shuffle=True,
        callbacks=[checkpointing, log_saving]
    )

    print(f'TRAINIG OF {name} COMPLETED')
    print('------------------------------')

model.load_weights(name)

(loss, acc) = model.evaluate(valid_generator, steps=STEP_SIZE_VALID)

print("\n############### EVALUATION ##############\n")
print(f"EVALUATED LOSS: {loss} EVALUATED ACC: {acc}")

STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
test_generator.reset()
pred=model.predict(test_generator, steps=STEP_SIZE_TEST, verbose=1)
predicted_class_indices=np.argmax(pred,axis=1)

labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

filenames=test_generator.filenames
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})
results.to_csv(f"results_predictions_{tag}.csv",index=False)

df = pd.read_csv(f"results_predictions_{tag}.csv",sep=',')

h = 0
m = 0

for i in range(len(df)):
    if df.Filename[i].split('/')[0] == df.Predictions[i]:
        h += 1
    else:
        #print(f"MISS! Filename: {df.Filename[i]} -> Prediction: {df.Predictions[i]}")
        m += 1
print("\n############### Prediction ##############\n")
print(f"HITS: {h} MISS: {m} TOTAL: {h+m}")
print(f"Trefferquote: {(h/len(df))*100}%")

        
