import numpy as np
import pandas as pd
import pickle
from scipy import stats

def preprocess_dat(X_train_inp, start_i, end_i):
  
  def load_pickle(file_path):
      with open(file_path, "rb") as fil:
          return pickle.load(fil)

  pt_alien = load_pickle("/content/drive/MyDrive/AI-Project-3-data/model3-utils/powertrans.pkl")
  scaler_alien = load_pickle("/content/drive/MyDrive/AI-Project-3-data/model3-utils/mmscaler.pkl")
  pt_crew = load_pickle("/content/drive/MyDrive/AI-Project-3-data/model3-utils/powertranscrew.pkl")
  scaler_crew = load_pickle("/content/drive/MyDrive/AI-Project-3-data/model3-utils/mmscalercrew.pkl")
  alien_enc = load_pickle("/content/drive/MyDrive/AI-Project-3-data/model3-utils/alienautoenc.pkl")
  crew_enc = load_pickle("/content/drive/MyDrive/AI-Project-3-data/model3-utils/crewautoenc.pkl")

  def process_data(X_inp, pt, scaler, autoenc):
      data_transformed = pt.transform(X_inp)
      data_scaled = scaler.transform(data_transformed)
      data_reduced = autoenc.predict(data_scaled)
      df_belief_network = pd.DataFrame(data_reduced)
      df_belief_network = df_belief_network.fillna(-999)
      for i in df_belief_network:
          if df_belief_network[i].mean() < 0:
              df_belief_network = df_belief_network.drop(columns=i)
      z_scores = np.abs(stats.zscore(df_belief_network))
      df_belief_network = z_scores
      df_belief_network = df_belief_network.fillna(-999)
      for i in df_belief_network:
          if df_belief_network[i].mean() < 0:
              df_belief_network = df_belief_network.drop(columns=i)
      return np.array(df_belief_network)

  alien_red = process_data(X_train_inp[:, 2500:5000], pt_alien, scaler_alien, alien_enc)
  crew_red = process_data(X_train_inp[:, 5000:7500], pt_crew, scaler_crew, crew_enc)

  move_use = np.load("/content/drive/MyDrive/AI-Project-3-data/old data/final npy/pos_moves_X_train.npy")[
              start_i:end_i]
  b_pos = X_train_inp[:, -4:-2]
  X_train_processed = np.concatenate((alien_red, crew_red, b_pos, move_use), axis=-1)

  return X_train_processed
