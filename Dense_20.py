import tensorflow as tf
import pandas as pd
import numpy as np
import data_generator as dg
import random
import os
from matplotlib import pyplot
import time
rng=random.SystemRandom()
#rng=random.seed(1237981728729821)
auto_corr_list=[]


def initialize(rng):
    config=np.zeros((N,N),np.int)
    def f(prob):
        if prob<0.5:
            return np.array([0,1])
        else:
            return np.array([1,0])
    config=np.array([np.array([f(rng.random()) for k in range(N)]) for j in range(N)])
    return config


def conv_markov_chain(config,N,T,steps,rng,labeled=True,batch_size=16):
    M=np.zeros(steps)
    configs=np.array([np.array([np.array([[0,1] for k in range(N)]) for j in range(N)])  for k in range(steps)])
    #configs=[]
    def E(list):
        if list[0]==1:
            return -1
        else:
            return 1
    for i in range(steps):
        for nx in range(N):
            for ny in range(N):
                nx1=np.mod(nx-1,N)
                nx2=np.mod(nx+1,N)
                ny1=np.mod(ny-1,N)
                ny2=np.mod(ny+1,N)
                de=2*E(config[nx,ny])*(E(config[nx1,ny])+E(config[nx2,ny])+E(config[nx,ny1])+E(config[nx,ny2]))
                if rng.random()<np.exp(-de/T):
                    config[nx,ny][1],config[nx,ny][0]=config[nx,ny][0],config[nx,ny][1]
        M[i]=np.abs(sum(sum(np.array([[E(config[i,j]) for i in range(N)] for j in range(N)]))))
        M[i]/=N**2
        configs[i]=np.copy(config)
    auto_corr=dg.autocorrelation(np.copy(M))
    j=0
    for i in auto_corr:
        if i>=0.5:
            j+=1
        else:
            j*=10
            break

    auto_corr_list.append(j)
    print("j=",j)
    return [configs[j:],np.copy(config)]


def arr_to_ds(configs, T = [],split=None,labeled=True,batch_size=16):
    if labeled:
        labels=np.zeros(sum(split),dtype=np.int)
        now_index=0
        for i in range(len(T)):
            if i==0:
                if T[i]<=1e-2:
                    labels[:split[i]]=np.ones(split[i])
                elif T[i]>=4:
                    lables[:split[i]]=np.zeros(split[i])
                now_index+=split[i]
            elif i<=len(T)-1:
                if T[i]<=1e-2:
                    labels[now_index:now_index+split[i]]=np.ones(split[i],dtype=np.int)
                elif T[i]>=4:
                    labels[now_index:now_index+split[i]]=np.zeros(split[i],dtype=np.int)
                now_index+=split[i]
                print(i,now_index)
        print(labels.shape[0])
        ds=tf.data.Dataset.from_tensor_slices(
            (
                tf.cast(configs,tf.float32),
                tf.cast(labels,tf.int32)
            )
        )
        ds=ds.shuffle(buffer_size=sum(split)).batch(batch_size=batch_size).repeat()
    else:
        ds=tf.data.Dataset.from_tensor_slices(
            (
                tf.cast(configs,tf.float32)
            )
        )
        ds=ds.batch(batch_size=batch_size)
    return ds


def conv_to_dense(config):
    def Spin(elem):
        if type(elem)==int:
            return elem
        if len(elem)==1:
            return elem[0]
        if elem[0]==1:
            return -1
        elif elem[0]==0:
            return 1
        else:
            return (elem[0]**2+elem[1]**2)**0.5

    dense_config=np.array([[Spin(config[i,j]) for i in range(config.shape[1])] for j in range(config.shape[0])])
    dense_config=np.reshape(dense_config,dense_config.shape[0]*dense_config.shape[1])
    return dense_config


def plot_graph(T,prediction,auto_corr_list,model):
    pyplot.scatter(T, prediction)
    prediction = [1 - x for x in prediction]
    pyplot.scatter(T, prediction)
    pyplot.ylabel('output layer')
    pyplot.xlabel('Temperature(k)')
    pyplot.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    i = 0
    image_path = 'order_vs_T_20_'+model+'_static_random' + '_' + str(i) + '.png'
    while os.path.isfile(image_path):
        i += 1
        image_path = 'order_vs_T_20_'+model+'_static_random' + '_' + str(i) + '.png'
    pyplot.savefig(image_path)
    pyplot.show()
    i=0
    pyplot.scatter(T,auto_corr_list)
    image_path = 'autocorrection_vs_T_20_'+model+'_static_random' + '_' + str(i) + '.png'
    while os.path.isfile(image_path):
        i += 1
        image_path = 'autocorrection_vs_T_20_'+model+'_static_random' + '_' + str(i) + '.png'
    pyplot.savefig(image_path)
    pyplot.show()


if __name__=='__main__':
    N=20
    filepath = 'Conv_ordered.h5'
    if os.path.isfile('./train_20_ds.h5'):
        ds=pd.read_hdf('train_20_ds.h5')
        train_ds=ds.iloc[:int(ds['labels'].count()*0.8)]
        test_ds=ds.iloc[int(ds['labels'].count()*0.8):]
    else:
        start = time.time()
        N=20
        rng=random.SystemRandom()
        config=initialize(rng)
        ds_1=conv_markov_chain(config, N, 1e-3, 5000, rng, False)
        ds_1,config=ds_1[0],ds_1[1]
        ds_2=conv_markov_chain(config, N, 1e-4, 5000, rng, False)
        ds_2,config=ds_2[0],ds_2[1]
        ds_3=conv_markov_chain(config, N, 5, 10000, rng, False)
        ds_3,config=ds_3[0],ds_3[1]
        df=np.append(ds_1,ds_2,axis=0)
        df=np.append(df,ds_3,axis=0)
        end=time.time()
        print('data generated')
        print(str(end-start), 'second is needed for data generation')

    df = np.array([conv_to_dense(df[i])for i in range(df.shape[0])])
    ds = arr_to_ds(df, T=[1e-3,1e-4,5], split=[ds_1.shape[0], ds_2.shape[0], ds_3.shape[0]])
    train_ds = ds.take(int(df.shape[0]*0.8))
    test_ds = ds.skip(int(df.shape[0]*0.2))
    pred = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128,activation=tf.nn.relu, input_shape=(400,)),
        tf.keras.layers.Dense(64,activation=tf.nn.relu),
        tf.keras.layers.Dense(32,activation=tf.nn.relu),
        tf.keras.layers.Dense(16,activation=tf.nn.relu),
        tf.keras.layers.Dense(1,activation=tf.nn.sigmoid, input_shape=(16,))]
    )

    pred.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001,epsilon=1e-7),
                 loss='binary_crossentropy', metrics=['accuracy'])
    start = time.time()
    pred.fit(train_ds,epochs=5,steps_per_epoch=1000)
    end=time.time()
    print(str(end - start), 'second is needed for model training')
    pred.evaluate(test_ds,steps=100)
    tf.keras.models.save_model(pred,'predictor.h5')

    def forward(input, predictor=pred):
        return predictor.predict(np.array([conv_to_dense(input[j]) for j in range(input.shape[0])]))


    def transition_temperature_searcher(T=(2.2, 2.34), configuation=config, forward_function=forward, N=20):
        transition = np.average(T)
        transition_config = conv_markov_chain(configuation, N, transition, 300, rng, labeled=False)
        prob_of_disorder=np.average(np.abs(transition_config[0]))
        if np.abs(prob_of_disorder-0.5)<1e-8:
            return transition
        elif prob_of_disorder>0.5:
            return transition_temperature_searcher(T=[transition,T[1]],configuation=transition_config[1],
                                                   forward_function=forward_function, N=20)
        else:
            return transition_temperature_searcher(T=[T[0],transition], configuation=transition_config[1],
                                                   forward_function=forward_function, N=20)

    print(np.average(np.abs(pred.predict(np.array([conv_to_dense(ds_1[i]) for i in range(ds_1.shape[0])])))))
    print(np.average(np.abs(pred.predict(np.array([conv_to_dense(ds_3[i]) for i in range(ds_3.shape[0])])))))

    auto_corr_list=[]
    T=[1e-5+i*1e-1 for i in range(40)]
    for i in range(40):
        T.append(2+i*0.025)
    T.sort()
    ds=[]
    for i in range(80):
        temp=conv_markov_chain(config,N,T[i],300,rng,False)
        config=np.copy(temp[1])
        ds.append(np.copy(temp[0]))
    ds=[np.array([conv_to_dense(ds[i][j]) for j in range(ds[i].shape[0])]) for i in range(len(ds))]
    print('predicting')
    prediction=[np.average(np.abs(pred.predict(ds[i]))) for i in range(len(ds))]
    print(prediction)
    plot_graph(T, prediction, auto_corr_list, model="dense")

    start=time.time()
    print(transition_temperature_searcher())
    end=time.time()
    print(str(end - start), 'second is needed for transition temperature searching')