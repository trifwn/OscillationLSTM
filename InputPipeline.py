import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from profiling import simple_timer
from pltfigure import pltfigure


class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
                 train_df, val_df, test_df,batch_size,
                 md_train_df = None, md_test_df = None,md_val_df = None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.batch_size = batch_size

        #Store Metadata
        self.md_train_df = md_train_df
        self.md_test_df = md_test_df
        self.md_val_df = md_val_df

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[
            self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[
            self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}'])

    def split_window(self, data):
        inputs = tf.stack(
            [data[:, self.input_slice, i] for i in range(0, data.shape[2])], axis=0
        )
        labels = tf.stack(
            [data[:, self.labels_slice, i] for i in range(0, data.shape[2])], axis=0
        )

        shape = [tf.shape(inputs)[k] for k in range(3)]
        inputs = tf.reshape(inputs, [shape[0]*shape[1], shape[2], 1])
        shape = [tf.shape(labels)[k] for k in range(3)]
        labels = tf.reshape(labels, [shape[0]*shape[1], shape[2], 1])

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, 1])
        labels.set_shape([None, self.label_width, 1])

        return inputs, labels

    def add_metadata(self,metadata,data):
        (inputs ,labels )= data
        metadata = tf.expand_dims(metadata,axis=-1)
        metadata = metadata[:tf.shape(inputs)[0],:,:]
        inputs = tf.concat([inputs,metadata],axis=1)
        return  inputs , labels

    def make_dataset(self, data, added_data = None, batch_size = 10):
        data = np.array(data, dtype=np.float32)
        added_data = np.array(added_data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=1) # Change Batch size for efficiency
        ds = ds.map(self.split_window)
        ds = ds.unbatch().batch(batch_size)
        
        if self.md_train_df is not None:
            try:
                ds2 = tf.data.Dataset.from_tensor_slices(added_data.T).repeat(-1)
                ds2 = ds2.batch(batch_size)
                ds = tf.data.Dataset.zip((ds2,ds))
                ds = ds.map(self.add_metadata)
            except:
                print("No added data")
        return ds
    
 
    @property
    def train(self):
        return self.make_dataset(self.train_df, added_data=self.md_train_df,batch_size= self.batch_size)

    @property
    def val(self):
        return self.make_dataset(self.val_df, added_data=self.md_val_df,batch_size= self.batch_size)

    @property
    def test(self):
        return self.make_dataset(self.test_df, added_data=self.md_test_df,batch_size= self.batch_size)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result

    def plotexample(self, model=None, plot_col='x', max_subplots=3):
        """Plot a single example batch of `inputs, labels`."""
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n+1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, self.input_slice, :],
                     label='Inputs', marker='.', zorder=-10)
            plt.scatter(self.label_indices, labels[n, :, :],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)

            if model is not None:
                predictions  = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :],
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Timesteps')

    def CalcCase(self, data, timesteps, model,verbose = False):
        inputsRT = data[:self.input_width]
        outputsRT = data[:self.input_width]
        
        iw = self.input_width
        shift = self.shift
        i=0
        while(iw+shift*(i+1)<=timesteps.shape[0]):
            predRT = np.squeeze(model(np.reshape(inputsRT, (1, inputsRT.shape[0], 1))))
            outputsRT = np.hstack([outputsRT, predRT])
            inputsRT = outputsRT[-iw:]
            i+=1
        if verbose ==True:
            print("We had to execute {} calls\nPredicted {} timesteps".format(i,np.shape(outputsRT[iw:])[0]))
        return  outputsRT
 
    def plotCase(self, data, timesteps,metadata=None, model=None, options = {"showLines" : True,  "yLabel" :'x', "xLabel" :'Timesteps'}):
        # inputs = train_df[:,index]
        inputsRT = data[:self.input_width]
        outputsRT = data[:self.input_width]

        if metadata is not None:
            inputsRT = np.hstack([inputsRT,metadata])
        
        iw = self.input_width
        shift = self.shift
        i=0
        
        plt.figure(figsize=(12, 8))
        plt.subplot(1, 1, 1)
        plt.ylabel(f'{options["yLabel"]} [normed]')
        plt.plot(self.input_indices, data[:self.input_width],
                 label='Inputs', marker='.', zorder=-10)

        while(iw+shift*(i+1)<=timesteps.shape[0]):
            label_indeces_new = [z+shift*i for z in self.label_indices]
            plt.scatter(label_indeces_new, data[label_indeces_new],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)

            if model is not None:
                predRT = np.squeeze(model(np.reshape(inputsRT, (1, inputsRT.shape[0], 1))))
                outputsRT = np.hstack([outputsRT, predRT])
                inputsRT = outputsRT[-iw:]
                if metadata is not None:
                    inputsRT = np.hstack([inputsRT,metadata])
                plt.scatter(label_indeces_new, predRT,
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)
            i+=1
            if options["showLines"] == True:
                plt.axvline(x=label_indeces_new[0],label='_nolegend_')

        if model is not None:
            plt.legend(["input","target",'Predictions'])
        else:
            plt.legend(["input","target"])
            return None  

        plt.xlabel('Timesteps')
        plt.xlim(left=0)
        plt.show()      
        return  outputsRT

"""    
    def animate(self, dataset, timesteps,skipping=0, model=None, plot_col='x'):
        # data = train_df[:,index]
        DATA = []
        for data in dataset.T:
            inputsRT = data[:self.input_width]
            outputsRT = data[:self.input_width]
            
            iw = self.input_width
            shift = self.shift
            i=0
            if model is not None:
                while(iw+shift*(i+1)<timesteps.shape[0]):
                    predRT = np.squeeze(model(np.reshape(inputsRT, (1, inputsRT.shape[0], 1))))
                    outputsRT = np.hstack([outputsRT, predRT])
                    inputsRT = outputsRT[-iw:]
                    i+=1
                DATA.append(outputsRT)
            else:
                zeros =  np.zeros((len(timesteps)))
                DATA.append(zeros)
        DATA = np.array(DATA)
        print(np.shape(dataset),np.shape(DATA))
        if model is not None:
            pltfigure(dataset,DATA,timesteps,"Prediction","Target",f"outcome_{model.name}")
        else:
            pltfigure(dataset,DATA,timesteps,"Prediction","Target",f"Dataset")"""

class DataPipeline():
    def __init__(self,Data,Metadata):
        self.Data = Data
        self.Metadata = Metadata
        self.num_cases = Data.shape[1]
    
    def splitData(self,trainpercentage=0.8,validationpercentage=0.1,testpercentage=0.1):

        trainslice = slice(0,int(trainpercentage*self.num_cases))
        valslice = slice(int(trainpercentage*self.num_cases),int((trainpercentage+validationpercentage)*self.num_cases))
        testslice = slice(int((trainpercentage+validationpercentage)*self.num_cases),self.num_cases)
        self.train_df = self.Data[:, trainslice]
        self.meta_train_df = self.Metadata.T[:, trainslice]

        self.val_df = self.Data[:,valslice]
        self.meta_val_df = self.Metadata.T[:,valslice]

        self.test_df = self.Data[:,testslice]
        self.meta_test_df = self.Metadata.T[:,testslice]

    def normalizeData(self):
        train_mean = self.train_df.mean()
        train_std = self.train_df.std()

        self.train_df = (self.train_df - train_mean) / train_std
        self.val_df = (self.val_df - train_mean) / train_std
        self.test_df = (self.test_df - train_mean) / train_std



'''SIMPLE PIPELINE NO TENSORFLOW'''

class SimplePipeline():
    def __init__(self, num_samples, n_steps,xshape,val_cases,use_metadata=True):
        self.num_samples = num_samples
        self.n_steps = n_steps
        self.xshape = xshape
        self.use_metadata = use_metadata
        self.val_cases = val_cases

    def split_sequence(self,sequence):
        X, y = list(), list()
        for i in range(len(sequence)):
            # find the end of this pattern
            end_ix = i + self.n_steps
            # check if we are beyond the sequence
            if end_ix > len(sequence) - 1:
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return X, y

    def makeData(self,data,metadata):
        if self.use_metadata:
            x_inp = np.zeros((int((self.xshape - self.n_steps) * self.num_samples), self.n_steps + 6))
        else:
            x_inp = np.zeros((int((self.xshape - self.n_steps) * self.num_samples), self.n_steps))
        y_out = np.zeros((int((self.xshape - self.n_steps) * self.num_samples)))
        n_samp = int((self.xshape - self.n_steps))

        for n in range(data.shape[0]):
            if self.use_metadata:
                a = metadata[n,:]
                x_inp[n * n_samp:(n + 1) * n_samp, :self.n_steps], y_out[n * n_samp:(n + 1) * n_samp] = self.split_sequence(data[n, :])
                x_inp[n * n_samp:(n + 1) * n_samp, self.n_steps:] = a
            else:
                x_inp[n * n_samp:(n + 1) * n_samp, :], y_out[n * n_samp:(n + 1) * n_samp] = self.split_sequence(data[n, :])
        return x_inp, y_out

    def plotAlldata(self,data,x):
        for n in range(data.shape[0]):
            plt.plot(x, data[n, :])
        plt.grid()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

    def split_data(self,x_inp, y_out):
        split = (self.num_samples - self.val_cases)/self.num_samples
        x_inp_train = x_inp[:int(split*x_inp.shape[0]),:]
        y_out_train = y_out[:int(split*y_out.shape[0])]
        x_inp_val = x_inp[int(split*x_inp.shape[0]):,:]
        y_out_val = y_out[int(split*y_out.shape[0]):]
        return x_inp_train, y_out_train, x_inp_val, y_out_val


'''
Replacing This code with the above code
x = np.linspace(0., 160., num=xshape)
data = np.zeros((num_samples, xshape))
if use_a:
    x_inp = np.zeros((int((xshape - n_steps) * num_samples), n_steps + 6))
else:
    x_inp = np.zeros((int((xshape - n_steps) * num_samples), n_steps))
y_out = np.zeros((int((xshape - n_steps) * num_samples)))
n_pat = int((xshape - n_steps) * num_samples)
n_samp = int((xshape - n_steps))
np.random.seed(1)
for n in range(num_samples):
    a1 = np.random.rand(1)
    a2 = np.random.rand(1)
    a3 = np.random.rand(1)
    a4 = np.random.rand(1)
    a5 = np.random.rand(1)
    a6 = np.random.rand(1)
    data[n, :] = a4 * np.sin(a1 * x) + a5 * np.cos(a2 * x) + np.sin(a3 * x) + a6
    if use_a:
        a = np.array([a1, a2, a3, a4, a5, a6])
        metadata.append(a)
        a_tile = np.transpose(np.tile(a, n_samp))
        x_inp[n * n_samp:(n + 1) * n_samp, :n_steps], y_out[n * n_samp:(n + 1) * n_samp] = split_sequence(data[n, :],
                                                                                                          n_steps)
        x_inp[n * n_samp:(n + 1) * n_samp, -6:] = a_tile
    else:
        x_inp[n * n_samp:(n + 1) * n_samp, :], y_out[n * n_samp:(n + 1) * n_samp] = split_sequence(data[n, :], n_steps)
    plt.plot(x, data[n, :])
plt.grid()
plt.xlabel("x")
plt.ylabel("y")
plt.show()
metadata = np.squeeze(np.array(metadata))'''        