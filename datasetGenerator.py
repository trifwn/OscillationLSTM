import numpy as np
from pltfigure import pltfigure

#This is parameter space
# m*x''+c*x'+mx = 0
# => x'' + 2jwx' +w^2 x =0 , j = c/(2wm) , w = sqrt(k/m)
w = np.linspace(0.01, 1, 10)
z = np.linspace(0.01, 1, 10)
x0 = 1
v0 = 0

t = np.linspace(0.1, 100, 400)


inputData = np.array(np.meshgrid(w,z,x0,v0)).T.reshape(-1, 4)
print(np.shape(inputData))

def finalsolution(W,Z,X0,V0,T):
    # final sol is x = e^(-zwt) (x0cos(wnt)+ ((v0+z*wn*x0)/wn)*sin(wnt)) wn = w sqrt(1-z**2)
    WN = W * np.sqrt(1-Z**2)
    try:
        res = np.exp(-Z*W*T) * (X0*np.cos(WN*T) + ((V0 + Z *WN *X0)/WN)*np.sin(WN *T))
    except RuntimeWarning as e:
        print(WN)
    finally:
        return res 

x = list()

for case in inputData:
    x.append(finalsolution(case[0],case[1],case[2],case[3],t))
Data = np.array(x)
Data = np.swapaxes(Data,0,1)

print(np.shape(Data),np.shape(inputData),np.shape(t))

np.savetxt("Data/time1.csv", t, delimiter=",")
np.savetxt("Data/OutputData1.csv",Data,delimiter=",")
np.savetxt("Data/InputData1.csv",inputData,delimiter=",")

zeroline = np.zeros((np.shape(Data)[1],len(t)))
print(np.shape(Data.T),np.shape(zeroline))
pltfigure(zeroline,Data.T,t,"Dataset","Zeroline",'Graphs/singleFreq.gif')