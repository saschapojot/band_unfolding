import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

from datetime import datetime


#band unfolding for ssh model


v=1
w=2


N=200

M=int(N/2)

supKValsAll=[2*np.pi/(2*M)*m for m in range(0,M)]


def hs(K):
    """

    :param K: momentum in SBZ
    :return: h_s matrix for supercell
    """
    ret=np.zeros((4,4),dtype=complex)

    ret[0,1]=v
    ret[0,3]=w*np.exp(-1j*2*K)
    ret[1,0]=v
    ret[1,2]=w
    ret[2,1]=w
    ret[2,3]=v
    ret[3,0]=w*np.exp(1j*2*K)
    ret[3,2]=v

    return ret



def wrapper4Sup(K):
    """

    :param K: momentum in SBZ
    :return: K, sorted eigenvalues and corresponding eigenvectors
    """
    vals,vecs=np.linalg.eigh(hs(K))
    inds=np.argsort(vals)
    sortedVals=[vals[i] for i in inds]
    sortedVecs=[vecs[:,i] for i in inds]
    return [K,sortedVals,sortedVecs]




def unfolding(KEigVecList):
    """

    :param KEigVecList: [K,eig,vec]
    :return: [k, unfolded eigenvalue]
    """
    K,E,z=KEigVecList
    z0,z1,z2,z3=z
    xi0=np.array([z0,z1])/np.linalg.norm(np.array([z0,z1]),ord=2)
    xi1=np.array([z2,z3])/np.linalg.norm(np.array([z2,z3]),ord=2)

    sgn=np.vdot(xi0,xi1)/np.exp(1j*K)

    if np.isclose(sgn,1,rtol=1e-04, atol=1e-06):
        k=K
    else:
        k=K+np.pi

    return [k,E]


procNum=48

pool0=Pool(procNum)

tEigStart=datetime.now()

retAllKValsVecs=pool0.map(wrapper4Sup,supKValsAll)

tEigEnd=datetime.now()

print("eig time: ",tEigEnd-tEigStart)

listIntoUnfolding=[]
for item in retAllKValsVecs:
    K, sortedVals, sortedVecs=item
    length=len(sortedVals)
    for j in range(0,length):
        listIntoUnfolding.append([K,sortedVals[j],sortedVecs[j]])


pool1=Pool(procNum)

tUnfoldingStart=datetime.now()

retUnfolded=pool1.map(unfolding,listIntoUnfolding)

tUnfoldingEnd=datetime.now()

print("unfolding: ", tUnfoldingEnd-tUnfoldingStart)

unfoldedk=[]
unfoldedE=[]



for item in retUnfolded:
    k,E=item
    unfoldedk.append(k)
    unfoldedE.append(E)

unfoldedk=np.array(unfoldedk)
primkValsAll=[2*np.pi/N*n for n in range(0,N)]
def hp(k):
    """

    :param k: 0 to 2pi
    :return: target spectrum
    """
    ret=np.zeros((2,2),dtype=complex)
    ret[0,1]=v+w*np.exp(-1j*k)
    ret[1,0]=v+w*np.exp(1j*k)

    return ret

def wrapper4Prim(k):
    vals,vecs=np.linalg.eigh(hp(k))
    return [k,vals,vecs]


pool2=Pool(procNum)
tPrimEigStart=datetime.now()

retPrim=pool2.map(wrapper4Prim,primkValsAll)
tPrimEigEnd=datetime.now()

print("time for primitive cells: ",tPrimEigEnd-tPrimEigStart)
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

plt.figure()
plt.scatter(unfoldedk/np.pi,unfoldedE,color="blue",label="unfolded spectrum",s=4)
plt.xlabel("$k/\pi$")
plt.ylabel("$E$")

labelPrimitive="targer band"
for row in primSortedEigVals:
    plt.plot(kPrimList/np.pi,row,color="magenta",label=labelPrimitive)
    labelPrimitive = "_nolegend_"


plt.legend()
plt.title("$v=$"+str(v)+", $w=$"+str(w))
plt.savefig("unfolding.png")