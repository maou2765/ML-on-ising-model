import tensorflow as tf
import pandas as pd
import numpy as np
import data_generator as dg
import random
import os
from matplotlib import pyplot
import time
from Dense_20 import plot_graph
#rng=random.SystemRandom()
rng=random.seed(1237981728729821)
auto_corr_list=[]


def initialize(rng):
    def f(prob):
        if prob<0.5:
            return np.array([0,1])
        else:
            return np.array([1,0])
    config = np.array([np.array([f(rng.random()) for k in range(N)]) for j in range(N)])
    return config


def conv_markov_chain(config,N,T,steps,rng,labeled=True,batch_size=16):
    M=np.zeros(steps)
    configs=np.array([np.array([np.array([[0,1] for _ in range(N)]) for _ in range(N)])  for _ in range(steps)])

    def E(elem):
        if elem[0] == 1:
            return -1
        else:
            return 1

    def flip(config, nx, ny):
        nx1 = np.mod(nx - 1, N)
        nx2 = np.mod(nx + 1, N)
        ny1 = np.mod(ny - 1, N)
        ny2 = np.mod(ny + 1, N)
        de = 2 * E(config[nx, ny]) * (E(config[nx1, ny]) + E(config[nx2, ny]) + E(config[nx, ny1]) + E(config[nx, ny2]))
        if rng.random() < np.exp(-de / T):
            config[nx, ny][1], config[nx, ny][0] = config[nx, ny][0], config[nx, ny][1]
        return np.copy(config[nx,ny])

    for i in range(steps):
        config=np.array([[flip(config,nx,ny) for nx in range(N)] for ny in range(N)])
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


def model_builder(kernel=[2,2], stride=[1,1], input_size=(20,20,2)):
    conf=list(zip(kernel,stride))
    model = tf.keras.models.Sequential()
    output_size=input_size[0]
    for i, ks in enumerate(conf):
        if i == 0:
            model.add(tf.keras.layers.Conv2D(2,ks[0], ks[1], activation=tf.nn.relu, input_shape=input_size))
        elif i == (len(kernel)-1):
            model.add(tf.keras.layers.Conv2D(2,ks[0], ks[1], activation=tf.nn.sigmoid))
        else:
            model.add(tf.keras.layers.Conv2D(2,ks[0], ks[1], activation=tf.nn.relu))
        output_size = (output_size-ks[0])/float(ks[1])+1
    conf=conf[::-1]
    for i, ks in enumerate(conf):
        if i==(len(kernel)-1):
            model.add(tf.keras.layers.Conv2DTranspose(2, int(20-(output_size-1)*ks[1]+1),
                                                      ks[1], activation=tf.nn.sigmoid))
        else:
            model.add(tf.keras.layers.Conv2DTranspose(2, ks[0], ks[1], activation=tf.nn.relu))
        output_size=ks[1]*(output_size-1)+ks[0]

    return model


if __name__=='__main__':
    N=20
    filepath = 'Conv_ordered.h5'
    if os.path.isfile('./train_20_ds.h5'):
        ds=pd.read_hdf('train_20_ds.h5')
        train_ds=ds.iloc[:int(ds['labels'].count()*0.8)]
        test_ds=ds.iloc[int(ds['labels'].count()*0.8):]
    else:
        print('getting data')
        start=time.time()
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
        print('data generated')
        end=time.time()
        print(str(end-start), 'second is needed for data generation')

    print('training start')
    start=time.time()
    encoder= model_builder([2,4,6],[1,2,2],(20,20,2))
    encoder_ds=tf.data.Dataset.from_tensor_slices(
        (
            tf.cast(df, tf.int32),
            tf.cast(df, tf.int32))
    ).shuffle(buffer_size=df.shape[0]).batch(16).repeat()
    encoder.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001,epsilon=1e-7),
                    loss=tf.losses.mean_squared_error,
                    metrics=['accuracy'])
    encoder.fit(encoder_ds, epochs=5, steps_per_epoch=1000)
    tf.keras.models.save_model(encoder, filepath)
    encoder.evaluate(encoder_ds, steps=1000)
    df = encoder.predict(tf.data.Dataset.from_tensor_slices(tf.cast(df, tf.int32)).batch(16))
    print(df.shape)
    end=time.time()
    encoder_training_time=end-start
    print(str(encoder_training_time), 'second is needed for the training of autoencoder')


    df = np.array([conv_to_dense(df[i])for i in range(df.shape[0])])
    ds = arr_to_ds(df, T=[1e-3,1e-4,5], split=[ds_1.shape[0], ds_2.shape[0], ds_3.shape[0]])
    train_ds = ds.take(int(df.shape[0]*0.8))
    test_ds = ds.skip(int(df.shape[0]*0.2))
    start = time.time()
    pred = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128,activation=tf.nn.relu, input_shape=(400,)),
        tf.keras.layers.Dense(64,activation=tf.nn.relu),
        tf.keras.layers.Dense(32,activation=tf.nn.relu),
        tf.keras.layers.Dense(16,activation=tf.nn.relu),
        tf.keras.layers.Dense(1,activation=tf.nn.sigmoid, input_shape=(16,))]
    )

    pred.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001,epsilon=1e-7),
                 loss='binary_crossentropy', metrics=['accuracy'])
    pred.fit(train_ds,epochs=5,steps_per_epoch=1000)
    pred.evaluate(test_ds,steps=100)
    tf.keras.models.save_model(pred,'predictor.h5')
    end=time.time()
    print(str(end - start), 'second is needed for predictor training')
    print(str(end - start + encoder_training_time), 'second is needed for model training')
    print('without autoencoder:', np.average(np.abs(pred.predict(np.array([conv_to_dense(ds_1[i]) for i in range(ds_1.shape[0])])))))
    print('without autoencoder:', np.average(np.abs(pred.predict(np.array([conv_to_dense(ds_3[i]) for i in range(ds_3.shape[0])])))))


    def forward(input, encoder_to_use=encoder, predictor=pred):
        reduced_input = encoder_to_use.predict(input)
        return predictor.predict(np.array([conv_to_dense(reduced_input[j]) for j in range(reduced_input.shape[0])]))


    def transition_temperature_searcher(T=(2.2, 2.34), configuation=config, forward_function=forward, N=20):
        transition = np.average(T)
        transition_config = conv_markov_chain(configuation, N, transition, 300, rng, labeled=False)
        prediction=np.average(np.abs(transition_config[0]))
        if np.abs(prediction-0.5)<1e-8:
            return transition
        elif prediction>0.5:
            return transition_temperature_searcher(T=[transition,T[1]],configuation=transition_config[1],
                                                   forward_function=forward_function, N=20)
        else:
            return transition_temperature_searcher(T=[T[0],transition], configuation=transition_config[1],
                                                   forward_function=forward_function, N=20)


    print('test for T=1e-5k')
    reduced_ds_1=encoder.predict(ds_1)
    print(np.average(
            pred.predict(
                np.array([conv_to_dense(reduced_ds_1[j]) for j in range(reduced_ds_1.shape[0])])
                )
            )
        )
    print('test for T=5k')
    reduced_ds_3 = encoder.predict(ds_3)
    print(np.average(
            pred.predict(
                np.array([conv_to_dense(reduced_ds_3[j]) for j in range(reduced_ds_3.shape[0])])
                )
            )
        )

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
    ds=[encoder.predict(item) for item in ds]
    print('dimensionality reduced')
    ds=[np.array([conv_to_dense(ds[i][j]) for j in range(ds[i].shape[0])]) for i in range(len(ds))]
    print('predicting')
    prediction=[np.average(np.abs(pred.predict(ds[i]))) for i in range(len(ds))]
    print(prediction)
    plot_graph(T,prediction,auto_corr_list, model="reduced")

    start = time.time()
    print(transition_temperature_searcher())
    end = time.time()
    print(str(end - start), 'second is needed for transition temperature searching')