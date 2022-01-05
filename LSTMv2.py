
# # Imports


from window import WindowGenerator
import os
import time
from datetime import datetime

from threading import Thread
import IPython
import IPython.display

import numpy as np
from math import *

import tensorflow as tf
from tensorflow.keras.layers import LSTM, LeakyReLU, Flatten, Dense
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

import matplotlib.pyplot as plt
from pltfigure import pltfigure


print(tf.__version__)


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
tf.test.is_built_with_cuda()


tf.executing_eagerly()


# # DATASET


InputData = np.genfromtxt('Data/InputData.csv', delimiter=",")
t = np.genfromtxt('Data/time.csv', delimiter=",")
Data = np.genfromtxt('Data/OutputData.csv', delimiter=",")
print(f"OutData: %s, \nInputData: %s, \nTime: %s" %
      (np.shape(Data), np.shape(InputData), np.shape(t)))


plt.plot(t, Data[:, 61])
plt.grid()


# # Split The Data


# You'll use a (70%, 20%, 10%) split for the training, validation, and test sets. Note the data is not being randomly shuffled before splitting. This is for two reasons:
#
#     It ensures that chopping the data into windows of consecutive samples is still possible.
#     It ensures that the validation/test results are more realistic, being evaluated on the data collected after the model was trained.
#


n = Data.shape[1]
train_df = Data[:, 0:int(n*0.7):]
val_df = Data[:, int(n*0.7):int(n*0.9):]
test_df = Data[:, int(n*0.9):]

num_features = Data.shape[1]


# # Normalize Data


train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std


example_ind = 61
plt.plot(t, train_df[:, example_ind])


# # Data Windowing


# ### Example of window


wExample = WindowGenerator(input_width=30, label_width=1,
                           shift=1, train_df=train_df, val_df=val_df, test_df=test_df)
wExample


# ### Example of slicing


# Stack three slices, the length of the total window.
example_window = tf.stack([np.array(train_df[:wExample.total_window_size, :]),
                           np.array(
                               train_df[100:100+wExample.total_window_size, :]),
                           np.array(train_df[200:200+wExample.total_window_size, :])])
example_inputs, example_labels = wExample.split_window(example_window)

print('Shapes are: (batch, time, features)')
print(f'Data shape: {example_window.shape}')
print(f'Inputs shape: {example_inputs.shape}')
print(f'Labels shape: {example_labels.shape}')


# ### Create tf.data.Datasets


wExample.train.element_spec


# ### Plot/Example


wExample.plotexample()
print(f'Inputs shape (batch, timesteps,features): {example_inputs.shape}')
print(f'Labels shape (batch, timesteps,features): {example_labels.shape}')


# # Plot Particular


OUT_STEPS = 30
wExample2 = WindowGenerator(input_width=30, label_width=OUT_STEPS,
                            shift=OUT_STEPS, train_df=train_df, val_df=val_df, test_df=test_df)


wExample2.plot(train_df[:, 63])


# # LSTM MODEL 1 TimeStep Train


def compile_and_fit(model, name, window, patience=10, MAX_EPOCHS=500, record=True):

    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y-%H:%M")

    NAME = name + "@"+str(MAX_EPOCHS)+"@"+dt_string
    filename = os.path.join("Models",  NAME + '.h5')

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')
    if record == True:
        tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
        checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=0,
                                     save_best_only=True, mode='min', save_weights_only=True)
        callbacks = [early_stopping, tensorboard, checkpoint]
    else:
        callbacks = [early_stopping]

    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=MAX_EPOCHS,
                        validation_data=window.val,
                        batch_size=16,
                        callbacks=callbacks)  # ,tensorboard,checkpoint])
    return history


val_performance = {}
performance = {}


lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time] => [batch, time, lstm_units]
    LSTM(256, activation='tanh', return_sequences=True, name="LSTM1"),
    LeakyReLU(alpha=0.4, name="LR1"),
    LSTM(256, activation='tanh', return_sequences=True, name="LSTM2"),
    LeakyReLU(alpha=0.4, name="LR2"),
    Flatten(name="Flat"),
    Dense(256, activation=None, use_bias=True, name="Dense1"),
    LeakyReLU(alpha=0.4, name="LR3"),
    Dense(units=1, name="Output")
])


windowOneStep = WindowGenerator(
    input_width=30, label_width=1, shift=1, train_df=train_df, test_df=test_df, val_df=val_df)
print('Input shape:', windowOneStep.example[0].shape)
print('Output shape:', lstm_model(windowOneStep.example[0]).shape)


windowOneStep


% % time
history = compile_and_fit(lstm_model, "LSTM_bigger",
                          windowOneStep, patience=10, record=False)

IPython.display.clear_output()

val_performance['LSTM'] = lstm_model.evaluate(windowOneStep.val)
performance['LSTM'] = lstm_model.evaluate(windowOneStep.test, verbose=0)


# # Plotting Results


# Example Plotting


windowOneStep.plotexample(lstm_model)


# Plotting Specific Case at Specific Time


startTime = 200
exampleCase = 31
CaseParameters = InputData[exampleCase]
print("W: {}\nZ: {}\nX0: {}\nV0: {}".format(*CaseParameters))
windowOneStep.plot(inputs=train_df[startTime:startTime +
                   windowOneStep.input_width+1, exampleCase], model=lstm_model)


# # Try to produce TimeSeries


def CalcCase(data, window, model):
    inputs = data[:window.input_width]
    inputsRT = data[:window.input_width]
    outputs = data[window.input_width:window.input_width+window.label_width]
    outputsRT = data[window.input_width:window.input_width+window.label_width]
    print(np.shape(inputs), np.shape(outputs))

    for step in t[window.input_width:]:

        pred = np.squeeze(model(np.reshape(inputs, (1, inputs.shape[0], 1))))
        predRT = np.squeeze(
            model(np.reshape(inputsRT, (1, inputs.shape[0], 1))))
        outputs = np.vstack([outputs, pred])
        outputsRT = np.vstack([outputsRT, predRT])

        inputs = data[step:step+window.input_width]
        inputsRT = outputsRT[-window.input_width:]
    return outputs, outputsRT


def plotCase(outputs, outputsRT, data, index):
    fig = plt.figure()
    axes = fig.add_axes([0, 0, 1, 1])
    axes.plot(t, outputs, label="Real Time Pred", linewidth=1.5)
    axes.plot(t, outputsRT, label="Input Pred", linewidth=1.5)
    axes.plot(t[:], data, label="Target", linestyle="dotted", linewidth=2)
    axes.grid()
    axes.legend()
    axes.set_title("Test Case")
    extent = axes.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig("Graphs/Agogos" + str(index) + '.png',
                bbox_inches=extent.expanded(1.2, 1.3),
                dpi=1000)
    plt.show()


testCase = train_df[:, 63]
out1, out2 = CalcCase(testCase, windowOneStep, lstm_model)
plotCase(out1, out2, testCase)
