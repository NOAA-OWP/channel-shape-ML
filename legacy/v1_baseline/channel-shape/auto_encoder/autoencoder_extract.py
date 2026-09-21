# Libraries 
# Catch warnings 
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import ae_data_loader as dataloader
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
import pandas as pd
import numpy as np
import os
import json
import sys 
from tqdm import tqdm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from keras import callbacks
from typing import Tuple

# --------------------------- Define auto encoder --------------------------- #
class AutoEncoderRef(Model):
  """ An autoencoder for the Refrence fabric features

  Parameters
  ----------
  in_size : int
      Number of neurons to expand 
  out_size : int
      Number of neurons to shrink to 
  dp: float
      Drop out rate
  act: str
      The activation function
  rand_state: int
      The random state
  
  Returns
  ----------
  None

  Example
  --------
  >>> sequential = AutoEncoderRef(in_size: 100, out_size: 10, dp: 0.5, act: 'relu', rand_state: 115)
        """
  def __init__(self, in_size: int, out_size: int, dp: float, act: str, rand_state: int) -> None:
    tf.random.set_seed(rand_state)
    super().__init__()
    self.encoder = keras.Sequential(
      [
        layers.Dense(in_size, kernel_initializer='glorot_uniform'),
        layers.BatchNormalization(),
        layers.Activation(act),
        layers.Dropout(dp),
        layers.Dense(np.round(float(in_size) / 2.0), kernel_initializer='glorot_uniform'),
        layers.BatchNormalization(),
        layers.Activation(act),
        layers.Dense(out_size, activation=act, kernel_initializer='glorot_uniform')
      ]
    )

    self.decoder = keras.Sequential(
      [
        layers.Dense(out_size, kernel_initializer='glorot_uniform'),
        layers.BatchNormalization(),
        layers.Activation(act),
        layers.Dropout(dp),
        layers.Dense(np.round(float(in_size) / 2.0), kernel_initializer='glorot_uniform'),
        layers.BatchNormalization(),
        layers.Activation(act),
        layers.Dropout(dp),
        layers.Dense(in_size, kernel_initializer='glorot_uniform'),
        layers.BatchNormalization(),
        layers.Activation(act),
        layers.Dense(in_size, activation="linear", kernel_initializer='glorot_uniform')
      ]
    )

  def call(self, inputs) -> keras.Sequential:
    encoded = self.encoder(inputs)
    decoded = self.decoder(encoded)
    return decoded

class AutoEncoderNLCD(Model):
  """ An autoencoder for the NLCD features

  Parameters
  ----------
  in_size : int
      Number of neurons to expand 
  out_size : int
      Number of neurons to shrink to 
  dp: float
      Drop out rate
  act: str
      The activation function
  rand_state: int
      The random state
  
  Returns
  ----------
  None

  Example
  --------
  >>> sequential = AutoEncoderNLCD(in_size: 100, out_size: 10, dp: 0.5, act: 'relu', rand_state: 115)
  """
  def __init__(self, in_size: int, out_size: int, dp: float, act: str, rand_state: int) -> None:
    tf.random.set_seed(rand_state)
    super().__init__()
    self.encoder = keras.Sequential(
      [
        layers.Dense(in_size, kernel_initializer='glorot_uniform'),
        layers.BatchNormalization(),
        layers.Activation(act),
        layers.Dropout(dp),
        layers.Dense(np.round(float(in_size) / 2.0), kernel_initializer='glorot_uniform'),
        layers.BatchNormalization(),
        layers.Activation(act),
        layers.Dropout(dp),
        layers.Dense(np.round(float(in_size) / 4.0), kernel_initializer='glorot_uniform'),
        layers.BatchNormalization(),
        layers.Activation(act),
        layers.Dense(out_size, activation=act, kernel_initializer='glorot_uniform')
      ]
    )

    self.decoder = keras.Sequential(
      [
        layers.Dense(out_size, kernel_initializer='glorot_uniform'),
        layers.BatchNormalization(),
        layers.Activation(act),
        layers.Dropout(dp),
        layers.Dense(np.round(float(in_size) / 4.0), kernel_initializer='glorot_uniform'),
        layers.BatchNormalization(),
        layers.Activation(act),
        layers.Dropout(dp),
        layers.Dense(np.round(float(in_size) / 2.0), kernel_initializer='glorot_uniform'),
        layers.BatchNormalization(),
        layers.Activation(act),
        layers.Dropout(dp),
        layers.Dense(in_size, kernel_initializer='glorot_uniform'),
        layers.BatchNormalization(),
        layers.Activation(act),
        layers.Dense(in_size, activation="linear", kernel_initializer='glorot_uniform')
      ]
    )

  def call(self, inputs) -> keras.Sequential:
    encoded = self.encoder(inputs)
    decoded = self.decoder(encoded)
    return decoded

class AutoEncoderLit(Model):
  """ An autoencoder for the lit features

  Parameters
  ----------
  in_size : int
      Number of neurons to expand 
  out_size : int
      Number of neurons to shrink to 
  dp: float
      Drop out rate
  act: str
      The activation function
  rand_state: int
      The random state
  
  Returns
  ----------
  None

  Example
  --------
  >>> sequential = AutoEncoderLit(in_size: 100, out_size: 10, dp: 0.5, act: 'relu', rand_state: 115)
  """
  def __init__(self, in_size: int, out_size: int, dp: float, act: str, rand_state: int) -> None:
    tf.random.set_seed(rand_state)
    super().__init__()
    self.encoder = keras.Sequential(
      [
        layers.Dense(in_size, kernel_initializer='glorot_uniform'),
        layers.BatchNormalization(),
        layers.Activation(act),
        layers.Dropout(dp),
        layers.Dense(np.round(float(in_size) / 2.0), kernel_initializer='glorot_uniform'),
        layers.BatchNormalization(),
        layers.Activation(act),
        layers.Dropout(dp),
        layers.Dense(np.round(float(in_size) / 4.0), kernel_initializer='glorot_uniform'),
        layers.BatchNormalization(),
        layers.Activation(act),
        layers.Dropout(dp),
        layers.Dense(np.round(float(in_size) / 8.0), kernel_initializer='glorot_uniform'),
        layers.BatchNormalization(),
        layers.Activation(act),
        layers.Dense(out_size, activation=act, kernel_initializer='glorot_uniform')
      ]
    )

    self.decoder = keras.Sequential(
      [
        layers.Dense(out_size, kernel_initializer='glorot_uniform'),
        layers.BatchNormalization(),
        layers.Activation(act),
        layers.Dropout(dp),
        layers.Dense(np.round(float(in_size) / 8.0), kernel_initializer='glorot_uniform'),
        layers.BatchNormalization(),
        layers.Activation(act),
        layers.Dropout(dp),
        layers.Dense(np.round(float(in_size) / 4.0), kernel_initializer='glorot_uniform'),
        layers.BatchNormalization(),
        layers.Activation(act),
        layers.Dropout(dp),
        layers.Dense(np.round(float(in_size) / 2.0), kernel_initializer='glorot_uniform'),
        layers.BatchNormalization(),
        layers.Activation(act),
        layers.Dropout(dp),
        layers.Dense(in_size, kernel_initializer='glorot_uniform'),
        layers.BatchNormalization(),
        layers.Activation(act),
        layers.Dense(in_size, activation="linear", kernel_initializer='glorot_uniform')
      ]
    )

  def call(self, inputs) -> keras.Sequential:
    encoded = self.encoder(inputs)
    decoded = self.decoder(encoded)
    return decoded

class Encode():
  """ A encoder model for all features

  Parameters
  ----------
  train_x : pd.DataFrame
      Number of neurons to expand 
  test_x : pd.DataFrame
      Number of neurons to shrink to 
  in_features: list
      A list containing names of all features to be encoded
  out_features: list
      A list containg names of encoded features 
  auto_encoder_obj: keras.Sequential
      A sequential model
  
  Returns
  ----------
  keras.Sequential
    A sequential model

  Example
  --------
  >>> sequential = Encode(train_x, test_x, in_features, out_features, auto_encoder_obj)
  """
  def __init__(self, train_x: pd.DataFrame, test_x: pd.DataFrame, 
               in_features: list, out_features: list,
                 auto_encoder_obj: keras.Sequential) -> None:
      self.train_x          = train_x
      self.test_x           = test_x
      self.in_features      = in_features
      self.out_features     = out_features
      self.auto_encoder_obj = auto_encoder_obj
      # Check directories
      if not os.path.isdir(os.path.join(os.getcwd(),"img/encoder_fit/")):
        os.mkdir(os.path.join(os.getcwd(),"img/encoder_fit/"))
      # Check directories
      if not os.path.isdir(os.path.join(os.getcwd(),"model/")):
        os.mkdir(os.path.join(os.getcwd(),"model/"))

  def plotFit(self, history: tf.keras.callbacks.History, 
              auto_encoder: keras.Sequential, train_x_sub: pd.DataFrame,
              test_x_sub: pd.DataFrame, key: str) -> None:
    
    def rsquared(obs: np.array, pred: np.array) -> float:
      """ Return R^2 where obs and pred are array-like."""
      slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(obs, pred)
      return r_value**2

    plt.clf()
    plt.figure()
    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title('model accuracy '+str(key))
    plt.ylabel('MAE loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('img/encoder_fit/'+str(key)+'_acc.png',bbox_inches='tight', dpi = 600, facecolor='white')
    plt.show()

    plt.clf()
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss MSE '+str(key))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('img/encoder_fit/'+str(key)+'_loss.png',bbox_inches='tight', dpi = 600, facecolor='white')
    plt.show()

    preds_train = pd.DataFrame(data=auto_encoder.predict(train_x_sub), columns=test_x_sub.columns)
    preds_test = pd.DataFrame(data=auto_encoder.predict(test_x_sub), columns=test_x_sub.columns)

    for indx in range(0, len(self.in_features)):
      try:
        print(train_x_sub.columns[indx])
        print("Training Accuracy: %.2f%%" % (rsquared(train_x_sub.iloc[:,indx], preds_train.iloc[:,indx])*100))
        print("Testing Accuracy: %.2f%%" % (rsquared(test_x_sub.iloc[:,indx], preds_test.iloc[:,indx])*100))
        plt.clf()
        plt.figure()
        plt.scatter(train_x_sub.iloc[:,indx], preds_train.iloc[:,indx])
        plt.xlabel('Observed') 
        plt.ylabel('Predicted') 
        plt.title("Training "+str(train_x_sub.columns[indx]))
        plt.savefig('img/encoder_fit/'+str(key)+'_'+str(train_x_sub.columns[indx])+'_train.png',bbox_inches='tight', dpi = 600, facecolor='white')
        plt.show()

        plt.clf()
        plt.figure()
        plt.scatter(test_x_sub.iloc[:,indx], preds_test.iloc[:,indx])
        plt.xlabel('Observed') 
        plt.ylabel('Predicted') 
        plt.title("Testing "+str(train_x_sub.columns[indx]))
        plt.savefig('img/encoder_fit/'+str(key)+'_'+str(train_x_sub.columns[indx])+'_test.png',bbox_inches='tight', dpi = 600, facecolor='white')
        plt.show()
        
        plt.clf()
        plt.figure()
        my_plot = plt.gcf()
        Y_max = test_x_sub.iloc[:,indx].max()
        Y_min = test_x_sub.iloc[:,indx].min()

        ax = sns.scatterplot(x= test_x_sub.iloc[:,indx], y=preds_test.iloc[:,indx])
        ax.set(ylim=(Y_min, Y_max))
        ax.set(xlim=(Y_min, Y_max))
        ax.set_xlabel("Predicted value of "+str(train_x_sub.columns[indx]))
        ax.set_ylabel("Observed value of "+str(train_x_sub.columns[indx]))

        X_ref = Y_ref = np.linspace(Y_min, Y_max, 100)
        plt.plot(X_ref, Y_ref, color='red', linewidth=1)
        plt.savefig('img/encoder_fit/'+str(key)+'_'+str(train_x_sub.columns[indx])+'_test_v2.png',bbox_inches='tight', dpi = 600, facecolor='white')
        my_plot = plt.gcf()
        plt.show()
      except:
        print("")

  def fitAutoEncoder(self, key: str, rand_state: int) -> Tuple[keras.Sequential, bool, str]:
    dp = 0.35
    act = 'selu'
    out_len = len(self.out_features)
    train_x_sub = self.train_x[self.in_features]
    test_x_sub = self.test_x[self.in_features]
    in_len = len(self.in_features)
    nan_flag = False
    nan_col = ""

    if np.isinf(train_x_sub).sum().sum() != 0:
      print("Found inf values check data transformation and input!")
    if train_x_sub.isna().sum().sum() != 0:
      print("Found Nan values creating dummy!")
      # ___________________________________________________
      # find last column containing nan
      temp = train_x_sub.isna().sum()
      col_name = temp[temp > 0].index[-1]
      train_x_sub['var_dummy'] = np.where(train_x_sub[col_name].isna(),0,1) 
      train_x_sub = train_x_sub.replace(np.nan, 0)

      temp = test_x_sub.isna().sum()
      col_name = temp[temp > 0].index[-1]
      test_x_sub['var_dummy'] = np.where(test_x_sub[col_name].isna(),0,1) 
      test_x_sub = test_x_sub.replace(np.nan, 0)
      in_len = in_len+1
      nan_flag = True
      nan_col = col_name
      
 
    auto_encoder = self.auto_encoder_obj(in_len, out_len, dp, act, rand_state)
    auto_encoder.compile(
        loss='mse',
        metrics=['mae'],
        optimizer='adam'
    )
    early_stopping = callbacks.EarlyStopping(monitor ="val_mae", 
                                        mode ="min", patience = 100, 
                                        restore_best_weights = True)
    model_checkpoint = callbacks.ModelCheckpoint('model/best_model_'+key+'.tf', 
                                                  monitor='val_mae', 
                                                  mode='min', verbose=2, 
                                                  patience = 100,
                                                  save_best_only=True)
    history = auto_encoder.fit(
        train_x_sub, 
        train_x_sub, 
        epochs=400, 
        batch_size=6, 
        validation_data=(test_x_sub, test_x_sub),
        callbacks =[early_stopping, model_checkpoint],
        verbose=0
    )
    # ___________________________________________________
    # Save
    # Recreate the exact same model purely from the file
    # new_model = keras.models.load_model('path_to_my_model')
    encoder = auto_encoder.layers[0]
    auto_encoder.save('model/autoencoder2_'+key, save_format='tf')
    encoder.save('model/encoder2_'+key, save_format='tf')
    # ___________________________________________________
    # Plot
    self.plotFit(history, auto_encoder, train_x_sub, test_x_sub, key)
    return encoder, nan_flag, nan_col

  def addFeature(self, org_data: pd.DataFrame, msk: np.array, 
                 train_x: pd.DataFrame, test_x: pd.DataFrame, 
                 encoder: keras.Sequential, in_features: list,
                 out_features: list, nan_flag: bool, nan_col: str) -> pd.DataFrame:
     # rebulid to orignal indexing of data for consitancy
      ind = pd.DataFrame(msk, columns=['mask'])
      train_id = ind.index[ind['mask'] == True].tolist()
      test_id = ind.index[ind['mask'] == False].tolist()
      train_x.index = train_id
      test_x.index = test_id
      data = pd.concat([train_x, test_x])

      # add new features
      if nan_flag:
        # find last column containing nan
        temp = data.isna().sum()
        data['var_dummy'] = np.where(data[nan_col].isna(),0,1) 
        data = data.replace(np.nan, 0)
        in_features.append('var_dummy')
      
      new_features = encoder.predict(data[in_features])
      for indx in range(0, len(out_features)):
        org_data[out_features[indx]] = new_features[:,indx]
      if nan_flag:
        in_features.remove('var_dummy')
      org_data = org_data.drop(in_features, axis=1)
      
      return org_data


# A driver class
class EncodeData:
  @staticmethod
  def main(argv):
      # ___________________________________________________
      # Input
      # os.chdir("..")
      print(os.getcwd())
      feature_extract = json.load(open('data/feature_extract_names.json'))
      rand_state = 105
      x_transform = True
      # ___________________________________________________
      # Bulid an instance of DataLoader object
      data_loader = dataloader.DataLoader(data_path='data/merged_fhg.parquet',
                                          rand_state=rand_state, 
                                          x_transform=x_transform) 
      org_data = data_loader.readFiles()
      msk = data_loader.splitData()
      train_x, test_x = data_loader.transformData()
      
      # ___________________________________________________
      # ref_fab
      key = 'ref_fab'
      in_features = feature_extract.get(key)
      out_features = feature_extract.get('ref_fab_extract')
      encode = Encode(train_x, test_x, in_features, out_features, AutoEncoderRef)
      encoder, nan_flag, nan_col = encode.fitAutoEncoder(key=key, rand_state=rand_state)
      org_data = encode.addFeature(org_data, msk, train_x, test_x, encoder, 
                                   in_features, out_features, nan_flag, nan_col)
     
      # ___________________________________________________
      # soil
      key = 'soil'
      in_features = feature_extract.get(key)
      out_features = feature_extract.get('soil_extract')
      encode = Encode(train_x, test_x, in_features, out_features, AutoEncoderRef)
      encoder, nan_flag, nan_col = encode.fitAutoEncoder(key=key, rand_state=rand_state)
      org_data = encode.addFeature(org_data, msk, train_x, test_x, encoder, 
                                   in_features, out_features, nan_flag, nan_col)
      
      # ___________________________________________________
      # stream_cat_hydraulic
      key = 'stream_cat_hydraulic'
      in_features = feature_extract.get(key)
      out_features = feature_extract.get('stream_cat_hydraulic_extract')
      encode = Encode(train_x, test_x, in_features, out_features, AutoEncoderRef)
      encoder, nan_flag, nan_col = encode.fitAutoEncoder(key=key, rand_state=rand_state)
      org_data = encode.addFeature(org_data, msk, train_x, test_x, encoder, 
                                   in_features, out_features, nan_flag, nan_col)

      # ___________________________________________________
      # stream_cat_lithology
      key = 'stream_cat_lithology'
      in_features = feature_extract.get(key)
      out_features = feature_extract.get('stream_cat_lithology_extract')
      encode = Encode(train_x, test_x, in_features, out_features, AutoEncoderLit)
      encoder, nan_flag, nan_col = encode.fitAutoEncoder(key=key, rand_state=rand_state)
      org_data = encode.addFeature(org_data, msk, train_x, test_x, encoder, 
                                   in_features, out_features, nan_flag, nan_col)

      # ___________________________________________________
      # stream_cat_ant
      key = 'stream_cat_ant'
      in_features = feature_extract.get(key)
      out_features = feature_extract.get('stream_cat_ant_extract')
      encode = Encode(train_x, test_x, in_features, out_features, AutoEncoderRef)
      encoder, nan_flag, nan_col = encode.fitAutoEncoder(key=key, rand_state=rand_state)
      org_data = encode.addFeature(org_data, msk, train_x, test_x, encoder, 
                                   in_features, out_features, nan_flag, nan_col)
      
      # ___________________________________________________
      # stream_cat_nlcd
      key = 'stream_cat_nlcd'
      in_features = feature_extract.get(key)
      out_features = feature_extract.get('stream_cat_nlcd_extract')
      encode = Encode(train_x, test_x, in_features, out_features, AutoEncoderNLCD)
      encoder, nan_flag, nan_col = encode.fitAutoEncoder(key=key, rand_state=rand_state)
      org_data = encode.addFeature(org_data, msk, train_x, test_x, encoder, 
                                   in_features, out_features, nan_flag, nan_col)
      
      # ___________________________________________________
      # flow_frequency
      key = 'flow_frequency'
      in_features = feature_extract.get(key)
      out_features = feature_extract.get('flow_frequency_extract')
      encode = Encode(train_x, test_x, in_features, out_features, AutoEncoderNLCD)
      encoder, nan_flag, nan_col = encode.fitAutoEncoder(key=key, rand_state=rand_state)
      org_data = encode.addFeature(org_data, msk, train_x, test_x, encoder, 
                                   in_features, out_features, nan_flag, nan_col)

      # ___________________________________________________
      # save
      org_data.to_parquet('data/Processed_merged_fhg.parquet')
      # org_data.to_csv('Processed_merged_fhg.csv', index=False)

      return

      
if __name__ == "__main__":
  EncodeData.main([])
    
