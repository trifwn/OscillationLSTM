from statistics import mode
import tensorflow as tf
from tensorflow.keras.layers import LSTM, LeakyReLU ,Flatten , Dense, Dropout
import numpy as np

class lstm_model(tf.keras.models.Sequential):
    def __init__(self,units =101,return_states = False,metadata = True):
        self.metadata = metadata
        super().__init__()   
        self.add(LSTM(units, activation='tanh', return_sequences=True,return_state=return_states,name = "LSTM1"))
        self.add(LSTM(units, activation='tanh', return_sequences=True,return_state=return_states,name = "LSTM2"))
        self.add(Flatten(name = "Flat"))
        self.add(Dense(units,activation=None, use_bias=True,name="Dense1"))
        self.add(Dense(units=1,activation = "linear" ,name="Output"))


class lstm_benchmark_model(tf.keras.models.Sequential):
    def __init__(self,units =64,return_states = False):
        super().__init__()       
        self.add(LSTM(units, activation='tanh', return_sequences=True,return_state=return_states,name = "LSTM1"))
        self.add(LSTM(units, activation='tanh', return_sequences=True,return_state=return_states,name = "LSTM2"))
        self.add(Flatten(name = "Flat"))
        self.add(Dense(units,activation=None, use_bias=True,name="Dense1"))
        self.add(LeakyReLU(alpha=0.3,name="LR1"))
        self.add(Dense(units,activation=None, use_bias=True,name="Dense2"))
        self.add(LeakyReLU(alpha=0.3,name="LR2"))
        self.add(Dense(units=1,activation = "linear" ,name="Output"))
    
class FeedBack(lstm_model):
    def __init__(self, units, out_steps):
        super().__init__(units = units,return_states = True)
        self.out_steps = out_steps
        self.units = units
 

    def warmup(self, inputs):
        x = inputs         
        carry_states = {}
        for layer in self.layers:
            if layer.__class__.__name__ == "LSTM":
                x, *carry_states[layer.name] = layer(x)
            else:
                x = layer(x)     

        return x , carry_states

    def call(self, inputs, carry_states = None, training=None, out_steps= None):
        
        if out_steps is None:
          out_steps = self.out_steps
  
        # Use a TensorArray to capture dynamically unrolled outputs.
        predictions = tf.cast(inputs,dtype=tf.float32)

        if carry_states is None:
            # Initialize the LSTM state.
            prediction, carry_states= self.warmup(inputs)
            # Insert the first prediction.
            predictions = tf.concat(axis=1, values = [predictions,tf.expand_dims(prediction,axis=2)]) # <<<< note the cast
            start =1 
        else:
            start = 0

        # Run the rest of the prediction steps.
        for n in range(start, out_steps):
            # Use the last prediction as input.
            x = predictions[:,n:,:]
            # Execute one lstm step.
            for layer in self.layers:
                if layer.__class__.__name__ == "LSTM":
                    x, *carry_states[layer.name] = layer(x, initial_state= [*carry_states[layer.name]],
                                                    training=training)
                else:
                    x = layer(x,training = training)  

            prediction = x

            predictions = tf.concat(axis=1, values = [predictions,tf.expand_dims(prediction,axis=2)]) # <<<< note the cast

            # predictions.shape => (time, batch, features)
        return predictions[:,-out_steps:,:]