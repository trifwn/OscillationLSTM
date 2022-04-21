import numpy as np
from pltfigure import pltfigure

num_samples = 28

np.random.seed(1)
a1 = np.random.rand(num_samples)
a2 = np.random.rand(num_samples)
a3 = np.random.rand(num_samples)
a4 = np.random.rand(num_samples)
a5 = np.random.rand(num_samples)
a6 = np.random.rand(num_samples)
x = np.linspace(0., 160., 400)


inputData = np.array(np.vstack((a1,a2,a3,a4,a5,a6))).T.reshape(-1, 6)
print(np.shape(inputData))

def finalsolution(A1,A2,A3,A4,A5,A6,T):
    try:
        res = A4 * np.sin(A1 * T) + A5 * np.cos(A2 * T) + np.sin(A3 * T) + A6
    except RuntimeWarning as e:
        print(a1,a2,a3,a4,a5,a6)
    finally:
        return res 

y = list()

for case in inputData:
    y.append(finalsolution(case[0],case[1],case[2],case[3],case[4],case[5],x))
Data = np.array(y)
Data = np.swapaxes(Data,0,1)

print(np.shape(Data),np.shape(inputData),np.shape(x))

np.savetxt("Data/timeBenchmark.csv", x, delimiter=",")
np.savetxt("Data/OutputDataBenchmark.csv",Data,delimiter=",")
np.savetxt("Data/InputDataBenchmark.csv",inputData,delimiter=",")

zeroline = np.zeros((np.shape(Data)[1],len(x)))
print(np.shape(Data.T),np.shape(zeroline))
pltfigure(zeroline,Data.T,x,"Dataset","Zeroline",'Graphs/GiannosDataset.gif')

