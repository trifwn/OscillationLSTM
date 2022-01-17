# %% [markdown]
# # Imports

# %%
import os
import time
from datetime import datetime

from threading import Thread
import IPython
import IPython.display

import numpy as np
from math import *

import tensorflow as tf
from tensorflow.keras.layers import LSTM, LeakyReLU ,Flatten , Dense,Dropout
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

import matplotlib.pyplot as plt
from pltfigure import pltfigure

# %%
print(tf.__version__)

# %%
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

# %%
tf.executing_eagerly()

# %% [markdown]
# # DATASET

# %%
InputData = np.genfromtxt('Data/InputData.csv',delimiter=",")
t = np.genfromtxt('Data/time.csv',delimiter=",")
Data = np.genfromtxt('Data/OutputData.csv',delimiter=",")
print(f"OutData: %s, \nInputData: %s, \nTime: %s" %(np.shape(Data),np.shape(InputData),np.shape(t)))

# %%
example_ind = 610

plt.plot(t,Data[:,example_ind])
plt.grid()

# %% [markdown]
# # Split The Data

# %% [markdown]
# You'll use a (70%, 20%, 10%) split for the training, validation, and test sets. Note the data is not being randomly shuffled before splitting. This is for two reasons:
# 
#     It ensures that chopping the data into windows of consecutive samples is still possible.
#     It ensures that the validation/test results are more realistic, being evaluated on the data collected after the model was trained.
# 

# %%
n = Data.shape[1]
train_df = Data[:, 0:int(n*0.7):]
val_df = Data[:,int(n*0.7):int(n*0.9):]
test_df = Data[:,int(n*0.9):]

num_features = Data.shape[1]


# %% [markdown]
# # Normalize Data

# %%
train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

# %%
plt.plot(t,train_df[:,example_ind])


# %% [markdown]
# # Data Windowing

# %%
from window import WindowGenerator

# %% [markdown]
# ### Example of window

# %%
wExample = WindowGenerator(input_width=30, label_width=1, shift=1,train_df=train_df,val_df=val_df,test_df=test_df)
wExample

# %% [markdown]
# ### Create tf.data.Datasets

# %%
wExample.train.element_spec

# %% [markdown]
# ### Plot/Example

# %%
wExample.plotexample()
print(f'Inputs shape (Batchsize, timesteps,features): {wExample.example[0].shape}')
print(f'Labels shape (Batchsize, timesteps,features): {wExample.example[1].shape}')


# %% [markdown]
# # Plot Particular

# %%
OUT_STEPS = 30
wExample2 = WindowGenerator(input_width=30, label_width=OUT_STEPS,
                            shift=OUT_STEPS, train_df=train_df, val_df=val_df, test_df=test_df)


# %%
wExample2.plot(train_df[:,63])

# %% [markdown]
# # LSTM MODEL 1 TimeStep Train

# %%
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
                        callbacks=callbacks)  # ,tensorboard,checkpoint])
    return history


val_performance = {}
performance = {}


# %%
lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time,1] => [batch, time, lstm_units]
    LSTM(64, activation='tanh', return_sequences=True,name = "LSTM1"),
    Dropout(0.2),
    LSTM(64, activation='tanh', return_sequences=True,name = "LSTM2"),
    Dropout(0.2),
    Flatten(name = "Flat"),
    Dense(64,activation=None, use_bias=True,name="Dense1"),
    LeakyReLU(alpha=0.4,name = "LR3"),
    Dense(units=1,activation = "linear" ,name="Output")
])

# %%
windowOneStep = WindowGenerator(
    input_width=30, label_width=1, shift=1,train_df=train_df,test_df=test_df,val_df=val_df)

print(windowOneStep)

# %%
windowOneStep.example[0].shape

# %%
print('Input shape:', windowOneStep.example[0].shape)
print('Output shape:', lstm_model(windowOneStep.example[0]).shape)

# %%
lstm_model.summary()

# %%
%%time
history = compile_and_fit(lstm_model, "LSTM",windowOneStep,MAX_EPOCHS=500 ,patience = 50,record = True)

IPython.display.clear_output()

val_performance['LSTM'] = lstm_model.evaluate(windowOneStep.val)
performance['LSTM'] = lstm_model.evaluate(windowOneStep.test, verbose=0)

# %% [markdown]
# # Plotting Results

# %% [markdown]
# Example Plotting

# %%
windowOneStep.plotexample(lstm_model)

# %% [markdown]
# Plotting Specific Case at Specific Time

# %%
startTime = 200
exampleCase = 310
CaseParameters = InputData[exampleCase]
print("W: {}\nZ: {}\nX0: {}\nV0: {}".format(*CaseParameters))
windowOneStep.plot( inputs=train_df[startTime:startTime + windowOneStep.input_width+1, exampleCase], model=lstm_model)

# %% [markdown]
# # Try to produce TimeSeries

# %%
def CalcCase(data, window, model):
    inputs = data[:window.input_width]
    inputsRT = data[:window.input_width]
    outputs = data[:window.input_width]
    outputsRT = data[:window.input_width]
    
    shift = window.shift
    for i, time in enumerate(t[window.input_width::shift]):
        pred = np.squeeze(model(np.reshape(inputs, (1, inputs.shape[0], 1))))
        predRT = np.squeeze(model(np.reshape(inputsRT, (1, inputsRT.shape[0], 1))))

        outputs = np.hstack([outputs, pred])
        outputsRT = np.hstack([outputsRT, predRT])

        inputs = data[i:i+window.input_width]
        inputsRT = outputsRT[-window.input_width:]
    return outputs, outputsRT

# %%
predData = []
predDataRT = []
for i in range(train_df.shape[1]):
    print(i)
    testCase = train_df[:,i]
    ts1 ,ts2 = CalcCase(testCase,windowOneStep,lstm_model)
    predData.append(ts1)
    predDataRT.append(ts2)
predData = np.array(predData)
predDataRT = np.array(predDataRT)


# %% [markdown]
# # Plot Individually

# %%
def plotCase(outputs, outputsRT, window, data, index):
    fig = plt.figure()
    axes = fig.add_axes([0, 0, 1, 1])
    iw = window.input_width

    axes.plot(t[:iw], data[:iw], label="Input", linewidth=1.5)
    axes.plot(t[iw:], data[iw:], label="Target", linestyle="dotted",linewidth=1.5)
    axes.plot(t[iw:], outputs[iw:], label="AR Predictions", 
                c='#2ca02c',linewidth = 1)
    axes.plot(t[iw:], outputsRT[iw:], label='Predictions',
                c='#ff7f0e',linewidth = 1)

    axes.grid()
    axes.legend()
    axes.set_title("Test Case")
    extent=axes.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig("Graphs/Agogos" + str(index) + '.png',
                dpi = 1000)
    plt.show()

# %%
ts1 ,ts2 = CalcCase(train_df[:,30],windowOneStep,lstm_model)
plotCase(ts1,ts2,windowOneStep,testCase,i)

# %%
# for ts1,ts2 in zip(predData,predDataRT):
#     plotCase(ts1,ts2,windowOneStep,testCase,i)

# %% [markdown]
# # Produce Animation

# %%
from pltfigure import pltfigure

# %%
pltfigure(train_df.T,predData,t,"Data","Prediction",'Graphs/datasetPred.gif')
pltfigure(train_df.T,predDataRT,t,"Data","Prediction",'Graphs/datasetPredRT.gif')

# %%



