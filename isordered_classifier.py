import tensorflow as tf
import pandas as pd
import numpy as np
import data_generator
import Machine_learning_model_FC
from os import urandom
import random
from matplotlib import pyplot
N=10
rng=random.SystemRandom()
df_base=pd.DataFrame(columns=[str(i)+'th element' for i in range(N*N)],copy=True)

def f(prob):
    if prob<0.5:
        return 1
    else:
        return -1
df=[i for i in range(80)]

config=np.zeros((N,N),np.int)
config=[[f(rng.random()) for i in config[j]] for j in range(N)]
config=np.array(config)
rng=random.SystemRandom()
T=[1e-8+i*1e-1 for i in range(40)]
for i in range(40):
    T.append(2+i*0.025)
T.sort()
loaded_model=tf.keras.models.load_model('./FC_ordered.h5')
'''
for i in range(80):
    config,_,_,_,df[i]=data_generator.markov_chain(config,N,T[i],600,rng,True,df_base)

ds=[Machine_learning_model_FC.df_to_ds(df[i],False,16,False) for i in range(80)]

prediction=[np.average(loaded_model.predict(ds[i],steps=10)) for i in range(80)]
print(prediction)
pyplot.scatter(T,prediction)
pyplot.savefig('order_vs_T.png')
pyplot.show()
'''
interval=[2.1,2.4]
while True:
    Transition=np.average(interval)
    print(Transition)
    config,_,_,_,df=data_generator.markov_chain(config,N,Transition,600,rng,True,df_base)
    df=Machine_learning_model_FC.df_to_ds(df,False,16,False)
    order=np.average(loaded_model.predict(df,steps=10))
    if np.abs(order-0.5)<1e-4 or np.abs(interval[1]-interval[0])<1e-4:
        print("The Transition temperature is ",Transition)
        break
    if order>0.5:
        interval[1]=Transition
        Transition=np.average(interval)
        print(interval)
        print('greater')
    elif order<0.5:
        interval[0]=Transition
        Transition=np.average(interval)
        print(interval)
        print('smaller')
