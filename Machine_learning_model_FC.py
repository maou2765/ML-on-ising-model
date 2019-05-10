import tensorflow as tf
import pandas as pd
import numpy as np
N=10
train_ds=pd.read_hdf('train_ds.h5')
test_ds=pd.read_hdf('test_ds.h5')
filepath='FC_ordered.h5'
def df_to_ds(df,shuffle=True,batch_size=32,labeled=True,repeat=True,N=10,conv=False):
    df=df.copy()
    if labeled:
        labels=df.pop('labels')
        if conv:
            matrix=np.array([[[df.iloc[k][str(i+j)+'th element'] for i in range(N)] for j in range(N)] for k in range(labels.count())])
            ds=tf.data.Dataset.from_tensor_slices(
                (
                    tf.cast(matrix,tf.float32),
                    tf.cast(labels.values,tf.float32)
                )
            )
        else:
            ds=tf.data.Dataset.from_tensor_slices(
            (
            tf.cast(df[[str(i)+'th element' for i in range(N*N)]].values,tf.float32),
            tf.cast(labels.values,tf.float32)
            )
            )
    else:
        if conv:
            matrix=np.array([[[df.iloc[k][str(i+j)+'th element'] for i in range(N)] for j in range(N)] for k in range(labels.count())])
            ds=tf.data.Dataset.from_tensor_slices(
            (
                tf,cast(matrix,tf.int32)
            )
            )
        else:
            ds=tf.data.Dataset.from_tensor_slices(
            (
            tf.cast(df[[str(i)+'th element' for i in range(N*N)]].values,tf.int32)
            )
            )
    if shuffle:
        ds=ds.shuffle(buffer_size=len(df))
    if repeat:
        ds=ds.batch(batch_size).repeat()
    return ds
if __name__=='__main__':
    train_ds=df_to_ds(train_ds,batch_size=16)
    test_ds=df_to_ds(test_ds,batch_size=16)
    model=tf.keras.models.Sequential([
    tf.keras.layers.Dense(64,activation=tf.nn.relu,kernel_regularizer=tf.keras.regularizers.l2(0.01),input_shape=(100,)),
    tf.keras.layers.Dense(32,activation=tf.nn.relu,kernel_regularizer=tf.keras.regularizers.l2(0.01),input_shape=(64,)),
    tf.keras.layers.Dense(16,activation=tf.nn.relu,kernel_regularizer=tf.keras.regularizers.l2(0.01),input_shape=(32,)),
    tf.keras.layers.Dense(1,activation=tf.nn.sigmoid,input_shape=(16,))
    ])

    model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001,epsilon=1e-7),loss='binary_crossentropy',metrics=['accuracy'])
    model.fit(train_ds,epochs=5,steps_per_epoch=1000)
    tf.keras.models.save_model(model,filepath)
    model.evaluate(test_ds,steps=100)
