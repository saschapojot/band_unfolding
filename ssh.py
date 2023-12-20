import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

from datetime import datetime


v=1
w=2

#band folding for ssh model

N=200

M=int(N/2)

primkValsAll=[2*np.pi/N*n for n in range(0,N)]

supKValsAll=[2*np.pi/(2*M)*m for m in range(0,M)]

def hp(k):
    ret=np.zeros((2,2),dtype=complex)
    ret[0,1]=v+w*np.exp(-1j*k)
    ret[1,0]=v+w*np.exp(1j*k)

    return ret

sigma1=np.zeros((2,2),dtype=complex)
sigma1[0,1]=-1j
sigma1[1,0]=1j

h1=np.kron(sigma1,sigma1)
def hs(K):
    ret=np.zeros((4,4),dtype=complex)

    ret[0,1]=v
    ret[0,3]=w*np.exp(-1j*2*K)
    ret[1,0]=v
    ret[1,2]=w
    ret[2,1]=w
    ret[2,3]=v
    ret[3,0]=w*np.exp(1j*2*K)
    ret[3,2]=v
    # ret+=0.1*h1

    return ret


def wrapper4Prim(k):
    vals,vecs=np.linalg.eigh(hp(k))
    return [k,vals,vecs]


def wrapper4Sup(K):
    vals,vecs=np.linalg.eigh(hs(K))
    return [K,vals,vecs]



procNum=48

pool0=Pool(procNum)
tPrimEigStart=datetime.now()

retPrim=pool0.map(wrapper4Prim,primkValsAll)
tPrimEigEnd=datetime.now()

print("time for primitive cells: ",tPrimEigEnd-tPrimEigStart)

pool1=Pool(procNum)
tSupEigStart=datetime.now()
retSup=pool1.map(wrapper4Sup,supKValsAll)

tSupEigEnd=datetime.now()

print("time for  supercells: ",tSupEigEnd-tSupEigStart)

#extract primitive cell's results

def list2Plot(retFromPool):
    #sort eigenvalues
    momentumList=[]
    length=len(retFromPool[0][1])
    sortedEigsList=[]
    for j in range(0,length):
        sortedEigsList.append([])
    for elem in retFromPool:
        momentum,vals,_=elem
        sortedVals=sorted(vals)
        momentumList.append(momentum)
        for j in range(0,length):
            sortedEigsList[j].append(sortedVals[j])
    return np.array(momentumList),sortedEigsList



kPrimList,primSortedEigVals=list2Plot(retPrim)

KSupList,supSortedEigVals=list2Plot(retSup)

labelPrimitive="primitive cell"

plt.figure()
for row in primSortedEigVals:
    plt.plot(kPrimList/np.pi,row,color="black",label=labelPrimitive)
    labelPrimitive = "_nolegend_"

labelSup="supercell"
for row in supSortedEigVals:
    plt.plot(KSupList/np.pi,row,color="red",label=labelSup)
    labelSup = "_nolegend_"

plt.legend()
plt.xlabel("momentum$/\pi$")
plt.ylabel("$E$")

plt.title("$v=$"+str(v)+", $w=$"+str(w))
plt.savefig("sshv"+str(v)+"w"+str(w)+".png")
plt.close()

