import numpy as np
import tensorflow as tf
import seaborn as sns
import random as rn
import os
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import metrics
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, MaxPooling1D, UpSampling1D,AveragePooling1D
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score

from CAE_Class import CAE


DatasetName='MP5'
data_inputs = pd.read_excel("Datasets\\"+ DatasetName+"_inputs.xlsx") #loading dataset
data_outputs=pd.read_csv("Datasets\outputs.csv") # loading outputs
TARGET=0
snc='x1'
inputs=data_inputs.to_numpy()
outputs=data_outputs.to_numpy()

train_input, test_input, train_output, test_output = train_test_split(inputs, outputs, test_size=0.25, random_state=7,shuffle=True)

latent_dim = 32
filter1=16
filter2=32
kernel_size_encoder=5
kernel_size_decoder=3
ep=20
batch=10
act="tanh"
act2="tanh"
denseNeuron=64

#ENCODER
encoder_inputs = keras.Input(shape=(700,1))

x = layers.Conv1D(filter1, kernel_size_encoder, activation=act,  padding="same")(encoder_inputs)
x = MaxPooling1D(pool_size = 2, padding='same')(x)

x = layers.Conv1D(filter2, kernel_size_encoder, activation=act,  padding="same")(x)
x = MaxPooling1D(pool_size = 2, padding='same')(x)

x = layers.Flatten()(x)

x = layers.Dense(denseNeuron, activation=act)(x)

H = layers.Dense(latent_dim, activation=act)(x)

encoder = keras.Model(encoder_inputs, H, name="encoder")
encoder.summary()

#DECODER
latent_inputs = keras.Input(shape=(latent_dim,))

x = layers.Dense(175 * 1 * filter2, activation=act2)(latent_inputs)
x = layers.Reshape((175, filter2))(x)

x = layers.Conv1DTranspose(filter2, kernel_size_decoder, activation=act2,  padding="same")(x)
x = UpSampling1D(2)(x)

x = layers.Conv1DTranspose(filter1, kernel_size_decoder, activation=act2,  padding="same")(x)
x = UpSampling1D(2)(x)

decoder_outputs = layers.Conv1DTranspose(1, kernel_size_decoder, activation=act2, padding="same")(x)


decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()



x_train = np.expand_dims(train_input, -1).astype("float32")
x_test = np.expand_dims(test_input, -1).astype("float32")
x_all = np.expand_dims(inputs[:,0:700], -1).astype("float32")

callback = tf.keras.callbacks.EarlyStopping(monitor='val_val_loss', patience=2, restore_best_weights=True)

cae = CAE(encoder, decoder)
cae.compile(optimizer=keras.optimizers.Adam(),loss='mse')
history=cae.fit(x_train, epochs=ep, batch_size=batch,validation_data=(x_test,x_test),callbacks=[callback],verbose=0)


# summarize history for loss
with plt.style.context(('seaborn-paper')):
    fig, ax = plt.subplots(figsize=(9, 5),dpi=300)
    ax.plot(history.history['loss'],label='training loss')
    ax.plot(history.history['val_val_loss'],label='validation loss')
    plt.xlabel('Epoch',fontsize=18)
    plt.ylabel('loss ',fontsize=18)
    ax.legend(fontsize=14)
    plt.show()
    fig.savefig("Paper7_figS7.svg", format="svg")

LatentVariables_train=cae.encoder.predict(x_train)
LatentVariables_test=cae.encoder.predict(x_test)
Decoded_Test=cae.decoder.predict(LatentVariables_test)
Decoded_Train=cae.decoder.predict(LatentVariables_train)


Results_R2Calibration = np.zeros((4,))
Results_R2Prediction = np.zeros((4))
Results_RMSE_Calibration = np.zeros((4))
Results_RMSE_Prediction = np.zeros((4))
prediction_tr= np.zeros((60,4))
prediction_ts= np.zeros((20,4))
for TARGET in range(outputs.shape[1]):
    
    ########################################################################################
    SonucNo="Sonuclar\\"+DatasetName+"_Sonuc"+str(snc)+"_Target"+str(TARGET)
    ########################################################################################
    ## LINEAR REGRESSION TRAINING
    lr = LinearRegression()
    lr.fit(LatentVariables_train, train_output[:,TARGET])
    prediction_tr[:,TARGET] = lr.predict(LatentVariables_train)
    prediction_ts[:,TARGET] = lr.predict(LatentVariables_test)
    
    score_tr = r2_score(train_output[:,TARGET], prediction_tr[:,TARGET])
    rmse_tr = mean_squared_error(train_output[:,TARGET], prediction_tr[:,TARGET], squared=False)
    
    score_ts = r2_score(test_output[:,TARGET], prediction_ts[:,TARGET])   
    rmse_ts = mean_squared_error(test_output[:,TARGET], prediction_ts[:,TARGET], squared=False)
    
    with plt.style.context(('seaborn-paper')):
        fig, ax = plt.subplots(figsize=(9, 5),dpi=300)
        #Plot the best fit line
        sns.regplot(x=prediction_tr[:,TARGET], y=train_output[:,TARGET],ci=None,color="b")
        #Plot the ideal 1:1 line
        ax.plot(train_output[:,TARGET], train_output[:,TARGET], color='green', linewidth=1)
        plt.title('$R^{2}$: '+str(score_tr))
        plt.xlabel('Predicted ')
        plt.ylabel('Measured ')
    
        plt.show()
        fig.savefig(SonucNo+"_train.png")
    
    with plt.style.context(('seaborn-paper')):
        fig, ax = plt.subplots(figsize=(9, 5),dpi=300)
        #Plot the best fit line
        sns.regplot(x=prediction_ts[:,TARGET], y=test_output[:,TARGET],ci=None,color="b")
        #Plot the ideal 1:1 line
        ax.plot(test_output[:,TARGET], test_output[:,TARGET], color='green', linewidth=2)
        plt.title('$R^{2}$: '+str(score_ts))
        plt.xlabel('Predicted ')
        plt.ylabel('Measured ')
    
        plt.show()
        fig.savefig(SonucNo+"_test.png")
        
    Results_R2Calibration[TARGET]=score_tr
    Results_R2Prediction[TARGET]=score_ts
    Results_RMSE_Calibration[TARGET]=rmse_tr
    Results_RMSE_Prediction[TARGET]=rmse_ts

results=np.concatenate((Results_R2Calibration.reshape((4,1)),Results_R2Prediction.reshape((4,1)), Results_RMSE_Calibration.reshape((4,1)),Results_RMSE_Prediction.reshape((4,1))),axis=1)
index_values = ['Target 0', 'Target 1', 'Target 2','Target 3'] 
column_values = ['R2Calibration', 'R2Prediction', 'RMSE_Calibration','RMSE_Prediction']
df1 = pd.DataFrame(data = results, index = index_values, columns = column_values)   

with pd.ExcelWriter(SonucNo+'_RESULTS.xlsx') as writer:  
    df1.to_excel(writer, sheet_name='All')

cae.save_weights(SonucNo+".h5")
tf.saved_model.save(cae, SonucNo)

