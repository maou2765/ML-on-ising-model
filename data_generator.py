import numpy as np
from os import urandom
import random
import math
from matplotlib import pyplot
import pandas as pd
def	energy(s):
  temp = 0
  N=s.shape[0]
  J=1
  temp += sum(sum(s[0:N-1,0:N]*s[1:N,0:N]))
  temp += sum(sum(s[0:N,0:N-1]*s[0:N,1:N]))
  temp += sum(s[0,0:N]*s[N-1,0:N])
  temp += sum(s[0:N,0]*s[0:N,N-1])
  return 2*temp*J*(-1)

def autocorrelation(ds):
    aver_M=sum(ds)/ds.size
    num_auto=int(0.1*ds.size)
    if num_auto>500:
        num_auto=500
    ds-=aver_M
    auto_corr=np.zeros(num_auto,np.float)
    for i in range(num_auto):
        auto_corr[i]=sum((ds[0:ds.size-i])*(ds[i:ds.size]))
    return auto_corr/auto_corr[0]
'''
def markov_chain(config,N,T,steps,rng,ds_gen=False,ds=None):
    E=energy(config)
    M=np.zeros(steps)
    ds=ds.copy()
    configs=[]
    for i in range(steps):
        for nx in range(N):
            for ny in range(N):
                nx1=np.mod(nx-1,N)
                nx2=np.mod(nx+1,N)
                ny1=np.mod(ny-1,N)
                ny2=np.mod(ny+1,N)
                de=2*config[nx,ny]*(config[nx1,ny]+config[nx2,ny]+config[nx,ny1]+config[nx,ny2])
                if rng.random()<np.exp(-de/T):
                    config[nx,ny]*=-1
                    E+=de
        M[i]=np.abs(sum(sum(config))/N**2)
        s=0
        config_dict={}
        if(ds_gen==True):
            for j in range(N):
                for k in range(N):
                    config_dict[str(s)+'th element']=[config[j,k]]
                    s+=1
            s=0
            if T<1e-2:
                config_dict['labels']=[1]
            elif T>3.99:
                config_dict['labels']=[0]
            configs.append(config_dict)
    auto_corr=autocorrelation(np.copy(M))
    if (ds_gen==True):
        j=0
        for i in auto_corr:
            if i>=0.5:
                j+=1
            else:
                j*=10
                break
        print(j,steps)
        if ds.empty:
            ds=pd.concat(objs=[pd.DataFrame(data=configs[l],copy=True) for l in range(j,steps)],ignore_index=True,copy=True,axis=0)
            print(ds.shape[-1])
        else:
            print(ds.head())
            df_list=[pd.DataFrame(data=configs[l],copy=True) for l in range(j,steps)]
            df_list.append(ds)
            print(len(df_list))
            ds=pd.concat(objs=df_list,ignore_index=True,copy=True,axis=0)
        print(ds.head())
    return config,E,M,auto_corr,ds
'''
def markov_chain(config,N,T,steps,rng,ds_gen=False,ds=None,labeled=True):
    E=energy(config)
    M=np.zeros(steps)
    ds=ds.copy()
    configs=[]
    for i in range(steps):
        for nx in range(N):
            for ny in range(N):
                nx1=np.mod(nx-1,N)
                nx2=np.mod(nx+1,N)
                ny1=np.mod(ny-1,N)
                ny2=np.mod(ny+1,N)
                de=2*config[nx,ny]*(config[nx1,ny]+config[nx2,ny]+config[nx,ny1]+config[nx,ny2])
                if rng.random()<np.exp(-de/T):
                    config[nx,ny]*=-1
                    E+=de
        M[i]=np.abs(sum(sum(config))/N**2)
        s=0
        config_dict={}
        if(ds_gen==True):
            for j in range(N):
                for k in range(N):
                    config_dict[str(s)+'th element']=[config[j,k]]
                    s+=1
            s=0
            if labeled:
                if T<1e-2:
                    config_dict['labels']=[1]
                elif T>3.99:
                    config_dict['labels']=[0]
            configs.append(config_dict)
    auto_corr=autocorrelation(np.copy(M))
    if (ds_gen==True):
        j=0
        for i in auto_corr:
            if i>=0.5:
                j+=1
            else:
                j*=10
                break
        if ds.empty:
            ds=pd.concat(objs=[pd.DataFrame(data=configs[l],copy=True) for l in range(j,steps)],ignore_index=True,copy=True,axis=0)
        else:
            df_list=[pd.DataFrame(data=configs[l],copy=True) for l in range(j,steps)]
            df_list.append(ds)
            print(len(df_list))
            ds=pd.concat(objs=df_list,ignore_index=True,copy=True,axis=0)
    return config,E,M,auto_corr,ds

def plotgraph(ds,T,epoch):
    auto_corr=[ac[3] for ac in ds]
    M=[np.abs(sum(m[2]))/epoch for m in ds]
    E=[e[1] for e in ds]
    for i in range(len(ds),5):
        pyplot.plot(range(200),auto_corr[i],'.',label=str(T[i]))
        pyplot.savefig(str(T[i])+'k\'s autocorrelation.png')
        pyplot.show()
    pyplot.scatter(T,M)
    pyplot.show()
    pyplot.scatter(T,E)
    pyplot.show()

rng=random.SystemRandom()
nT=30
T=np.array(range(nT),np.float)
M=np.array(range(nT),np.float)
E=np.array(range(nT),np.float)
auto_corr=[]
N=10
config=np.zeros((N,N),np.int)
def f(prob):
    if prob<0.5:
        return 1
    else:
        return -1
config=[[f(rng.random()) for i in config[j]] for j in range(N)]
config=np.array(config)
#test=markov_chain(config,3,N,1,rng)
epoch=2000

'''
configs=[]
for i in range(nT):
    T[i]=1+0.1*i
    configs.append(markov_chain(config,N,T[i],epoch,rng))
    E[i]=configs[-1][1]
    M[i]=np.abs(sum(configs[-1][2])/epoch)
    auto_corr.append(configs[-1][3])
'''
if __name__=='__main__':
    print('data generating')
    ds=pd.DataFrame(columns=[str(i)+'th element' for i in range(N*N)].append('labels'),copy=True)
    config,_,_,_,ds=markov_chain(config,N,1e-3,50000,rng,True,ds)
    config,_,_,_,ds=markov_chain(config,N,4.0,50000,rng,True,ds)
    np.random.shuffle(ds.values)
    print(ds['labels'].count())
    train_ds=ds.iloc[:int(ds['labels'].count()*0.8)]
    test_ds=ds.iloc[int(ds['labels'].count()*0.8):]
    train_ds_store=pd.HDFStore('train_ds.h5')
    test_ds_store=pd.HDFStore('test_ds.h5')
    train_ds_store['train']=train_ds
    test_ds_store['test']=test_ds
    print('dataset size:',ds['labels'].count())
    print('training set size:',train_ds['labels'].count())
    print('testing set size:',test_ds['labels'].count())
    print('dataset loading done')
