import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt
from datetime import datetime

v0=1
v1=2
v2=1.2
v3=4.3
Nk=3*30
kIndValsAll=[n for n in range(0,Nk)]
kValsAll=[2*np.pi*n/Nk for n in kIndValsAll]

#this script computes band-unfolding for L=4, N=3
def h0pEigs(j):
    """

    :param kj: momentum in BZ
    :return: [k, eigenvalues, eigenvectors]
    """
    k=kValsAll[j]
    mat=np.zeros((5,5),dtype=complex)
    mat[0,1]=v0
    mat[0,3]=np.exp(-1j*k)*np.conj(v3)

    mat[1,0]=np.conj(v0)
    mat[1,2]=v1

    mat[2,1]=np.conj(v1)
    mat[2,3]=v2

    mat[3,0]=np.exp(1j*k)*v3
    mat[3,2]=np.conj(v2)

    vals,vecs=np.linalg.eigh(mat)
    return [j,vals, vecs]



procNum=48
pool0=Pool(procNum)
eigStart=datetime.now()

retAll=pool0.map(h0pEigs,kIndValsAll)

eigEnd=datetime.now()
print("eig time: ",eigEnd-eigStart)
retSorted=sorted(retAll,key=lambda item: item[0])
#map k in BZ to K in SBZ, map eigenvalues

def ps(n):
    """

    :param n: index of k in BZ
    :return: map k in BZ to K in SBZ, along with group number
    """
    if 0<=3*n and 3*n<Nk:
        return [0,n]
    elif Nk<=3*n and 3*n<2*Nk:
        return [1,n-int(Nk/3)]
    else:
        return [2,n-2*int(Nk/3)]


def x2y(a,K,xvec):
    """

    :param a: group number, a=0,1,2
    :param K: momentum in SBZ
    :param xvec: eigenvector solved from h0p
    :return: map x eigenvector to y eigenvector
    """
    yvec=np.array(list(xvec)+list(xvec)+list(xvec))
    yvec[5:5+5]*=np.exp(1j*K)*np.exp(1j*2*np.pi*a/3)
    yvec[10:10+5]*=np.exp(1j*2*K)*np.exp(1j*2*np.pi*2*a/3)
    return yvec

supKEigsVecs=[]
#each entry of supKEigsVecs corresponds to one K in SBZ

for n in range(0,int(Nk/3)):
    _,vals0,vecs0=retSorted[n]
    K0=2*np.pi*n/Nk
    y0Mat=np.zeros((15,5),dtype=complex)
    a0=0
    for j in range(0,5):
        y0Mat[:,j]=x2y(a0,K0,vecs0[:,j])

    row0=[0,n,vals0,y0Mat]
    n1=n+int(Nk/3)
    _,vals1,vecs1=retSorted[n1]
    a1=1
    y1Mat=np.zeros((15,5),dtype=complex)
    for j in range(0,5):
        y1Mat[:,j]=x2y(a1,K0,vecs1[:,j])

    row1=[1,n1,vals1,y1Mat]
    n2=n+int(2*Nk/3)
    _,vals2,vecs2=retSorted[n2]
    a2=2
    y2Mat=np.zeros((15,5),dtype=complex)
    for j in range(0,5):
        y2Mat[:,j]=x2y(a2,K0,vecs2[:,j])

    row2=[2,n2,vals2,y2Mat]
    oneK=[row0,row1,row2]
    supKEigsVecs.append(oneK)

#plot band in SBZ

# KPlt=[]
# EigPlt=[]
# for i in range(0,15):
#     EigPlt.append([])
#
# for n in range(0,len(supKEigsVecs)):
#     K=2*np.pi*n/Nk
#     KPlt.append(K)
#     row0,row1,row2=supKEigsVecs[n]
#     eigsTmp=[]
#     for e in row0[2]:
#         eigsTmp.append(e)
#     for e in row1[2]:
#         eigsTmp.append(e)
#     for e in row2[2]:
#         eigsTmp.append(e)
#
#     eigsTmpSorted=sorted(eigsTmp)
#     for j in range(0,15):
#         EigPlt[j].append(eigsTmpSorted[j])
#
# KPlt=np.array(KPlt)
# plt.figure()
# for j in range(0,15):
#     plt.plot(KPlt/np.pi,EigPlt[j],color="black")
#
# plt.xlabel("$K/\pi$")
# plt.ylabel("Energy")
# plt.savefig("folded.png")
# plt.close()

V=np.zeros((5,5),dtype=complex)

V[0,1]=v0
V[1,0]=np.conj(v0)
V[1,2]=v1
V[2,1]=np.conj(v1)
V[2,3]=v2
V[3,2]=np.conj(v2)

U=np.zeros((5,5),dtype=complex)
U[3,0]=v3

W=np.zeros((5,5),dtype=complex)
u2=0.05
u3=0

W[2,4]=u2
W[3,4]=np.conj(u3)
W[4,2]=np.conj(u2)
W[4,3]=u3

def hs(n):
    """

    :param n: index of momentum in SBZ
    :return:[n,eigs, vecs]
    """
    K=kValsAll[n]
    r0=np.c_[V,U,np.exp(-1j*3*K)*np.conj(U.T)]
    r1=np.c_[np.conj(U.T),V,U]
    r2=np.c_[np.exp(1j*3*K)*U,np.conj(U.T),V]
    mat=np.r_[r0,r1,r2]
    vals,vecs=np.linalg.eigh(mat)
    return [n,vals,vecs]

supKIndAll=[n for n in range(0,Nk) if 3*n<Nk]
pool1=Pool(procNum)
tEigKStart=datetime.now()
retKAll=pool1.map(hs,supKIndAll)
tEigKEnd=datetime.now()
print("K eig time: ",tEigKEnd-tEigKStart)
retKSorted=sorted(retKAll,key=lambda item: item[0])

def groupValues(irow0row1row2):
    """

    :param irow0row1row2: one entry in supKEigsVecs
    :return:
    """
    i,row0,row1,row2=irow0row1row2
    #positions of eigenvalues and eigenvalues
    indexAndVals=[]
    for i in range(0,5):
        tmp0=[0,i,row0[2][i]]
        indexAndVals.append(tmp0)

        tmp1=[1,i,row1[2][i]]
        indexAndVals.append(tmp1)

        tmp2=[2,i,row2[2][i]]
        indexAndVals.append(tmp2)

    #sort eigenvalues
    sortedIndexAndVals=sorted(indexAndVals,key=lambda item: item[2])

    #group eigenvalues
    groups=[]
    epsr=1e-5
    epsa=1e-7
    groups.append([sortedIndexAndVals[0]])
    for i in range(1,len(sortedIndexAndVals)):
        lastVal=groups[-1][-1][2]
        entryCurr=sortedIndexAndVals[i]
        if np.isclose(lastVal,entryCurr[2],rtol=epsr,atol=epsa):
            groups[-1].append(entryCurr)
        else:
            groups.append([entryCurr])
    return [i,groups]


def zProjRow(z,row):
    """

    :param z: eigenvector of hs
    :param row: [a,n,vals,vecs]
    :return: abs of projection of z to each column vector in row
    """
    _,_,_,vecs=row
    coefs=[]
    rn,cn=vecs.shape
    zNormalized=z/np.linalg.norm(z,ord=2)
    for j in range(0,cn):
        vecTmpNormalized=vecs[:,j]/np.linalg.norm(vecs[:,j],ord=2)
        coefs.append(np.abs(np.vdot(zNormalized,vecTmpNormalized)))
    return np.array(coefs)



def projection(n):
    """

    :param n: index of K in SBZ
    :return:[n, [j,[inds0],[inds1],[inds2]]], j-th eigenvector of hs projected to each of the eigenvectors of h0s,
    inds0, inds1, inds2 are indices of non-zero projections
    """
    eps=1e-7
    _,valshs,vecshs=retKSorted[n]
    row0,row1,row2=supKEigsVecs[n]

    projInd=[n,[]]
    for j in range(0,len(valshs)):
        jInd=[j]
        zTmp=vecshs[:,j]

        coefs0Tmp=zProjRow(zTmp,row0)
        inds0Tmp=np.where(coefs0Tmp>eps)[0]
        jInd.append(inds0Tmp)

        coefs1Tmp=zProjRow(zTmp,row1)
        inds1Tmp=np.where(coefs1Tmp>eps)[0]
        jInd.append(inds1Tmp)

        coefs2Tmp=zProjRow(zTmp,row2)
        inds2Tmp=np.where(coefs2Tmp>eps)[0]
        jInd.append(inds2Tmp)
        projInd[-1].append(jInd)
    return projInd


pool2=Pool(procNum)

tProjStart=datetime.now()

retProjs=pool2.map(projection,supKIndAll)

tProjEnd=datetime.now()

print("projection time: ",tProjEnd-tProjStart)

sortedProjs=sorted(retProjs,key=lambda item: item[0])

kPlt0=[]
kPlt1=[]
kPlt2=[]

EPlt0=[]
EPlt1=[]
EPlt2=[]
for item in sortedProjs:
    n,listTmp=item
    for elem in listTmp:
        j,inds0,inds1,inds2=elem
        for elem in inds0:
            n0=n
            kPlt0.append(kValsAll[n0])
            EPlt0.append(retKSorted[n][1][j])
        for elem in inds1:
            n1=n+int(Nk/3)
            kPlt1.append(kValsAll[n1])
            EPlt1.append(retKSorted[n][1][j])
        for elem in inds2:
            n2=n+int(2*Nk/3)
            kPlt2.append(kValsAll[n2])
            EPlt2.append(retKSorted[n][1][j])
kPlt0=np.array(kPlt0)
kPlt1=np.array(kPlt1)
kPlt2=np.array(kPlt2)
plt.figure()
plt.scatter(kPlt0/np.pi,EPlt0,color="red")
plt.scatter(kPlt1/np.pi,EPlt1,color="red")
plt.scatter(kPlt2/np.pi,EPlt2,color="red")
plt.xlabel("$k/\pi$")
plt.ylabel("Energy")
plt.savefig("unfolded.png")
plt.close()