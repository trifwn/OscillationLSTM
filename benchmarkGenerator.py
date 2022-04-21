import numpy as np
from pltfigure import pltfigure

#This is parameter space
# m*x''+c*x'+mx = 0
# => x'' + 2jwx' +w^2 x =0 , j = c/(2wm) , w = sqrt(k/m)
a1 = np.random.rand(28)
a2 = np.random.rand(28)
a3 = np.random.rand(28)
a4 = np.random.rand(28)
a5 = np.random.rand(28)
a6 = np.random.rand(28)
t = np.linspace(0., 160., 400)


inputData = np.array(np.vstack((a1,a2,a3,a4,a5,a6))).T.reshape(-1, 6)
print(np.shape(inputData))

def finalsolution(A1,A2,A3,A4,A5,A6,T):
    # final sol is x = e^(-zwt) (x0cos(wnt)+ ((v0+z*wn*x0)/wn)*sin(wnt)) wn = w sqrt(1-z**2)
    try:
        res = A4 * np.sin(A1 * T) + A5 * np.cos(A2 * T) + np.sin(A3 * T) + A6
    except RuntimeWarning as e:
        print(a1,a2,a3,a4,a5,a6)
    finally:
        return res 

x = list()

for case in inputData:
    x.append(finalsolution(case[0],case[1],case[2],case[3],case[4],case[5],t))
Data = np.array(x)
Data = np.swapaxes(Data,0,1)

print(np.shape(Data),np.shape(inputData),np.shape(t))

np.savetxt("Data/timeBenchmark.csv", t, delimiter=",")
np.savetxt("Data/OutputDataBenchmark.csv",Data,delimiter=",")
np.savetxt("Data/InputDataBenchmark.csv",inputData,delimiter=",")

zeroline = np.zeros((np.shape(Data)[1],len(t)))
print(np.shape(Data.T),np.shape(zeroline))
pltfigure(zeroline,Data.T,t,"Dataset","Zeroline",'Graphs/GiannosDataset.gif')