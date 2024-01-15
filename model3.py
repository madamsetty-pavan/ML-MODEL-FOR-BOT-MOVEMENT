import numpy as np
import ast
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import to_categorical
import pandas as pd
import numpy as np
from keras.layers import Dense, Dropout, BatchNormalization
from google.colab import drive
from keras.models import Model
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

data = pd.read_excel("/Users/shauryajain/alien-rematch/project3/AI-Project3-Code/Model1 data.xlsx")
data.drop(4060)
ship = data['layout of ship ']
alien_prob = data['alien prob']
crew_prob = data['crew prob']
bot = data['position of bot']
move = data["next move"].tolist()

train_size = int((0.85)*len(data))
print("Data size: ", len(data))

X_train = []
y_train = []
X_test = []
y_test = []


for i in range(len(data)-1):
  a = ast.literal_eval(ship[i])
  b = list(ast.literal_eval(alien_prob[i][1:-1]))
  c = list(ast.literal_eval(crew_prob[i][1:-1]))
  d = ast.literal_eval(bot[i])

  X_train.append(a+b+c+d)
  y_train.append(float(move[i]))

X_train = np.array(X_train)
y_train = np.array(y_train)
y_train_cat = to_categorical(y_train, num_classes=4)

def preprocess_dat(X_train_inp, start_i, end_i):
  """
  Input : X_train_inp - A numpy array of shape (n,7504) i.e cosisting of n rows each of 7504 dimension i.e
          start_i - Start Index of npy_usage
          end_i   - End index of npy_usage
  Output: X_train_processed - A numpy array of shape (n,37)
  """

  import pickle
  from scipy import stats
  import numpy as np
  import pandas as pd


  with open("/Users/shauryajain/alien-rematch/project3/AI-Project3-Code/model3-utils/powertrans.pkl","rb") as fil:
    pt_alien = pickle.load(fil)

  with open("/Users/shauryajain/alien-rematch/project3/AI-Project3-Code/model3-utils/mmscaler.pkl","rb") as fil:
    scaler_alien = pickle.load(fil)

  with open("/Users/shauryajain/alien-rematch/project3/AI-Project3-Code/model3-utils/powertranscrew.pkl","rb") as fil:
    pt_crew = pickle.load(fil)

  with open("/Users/shauryajain/alien-rematch/project3/AI-Project3-Code/model3-utils/mmscalercrew.pkl","rb") as fil:
    scaler_crew = pickle.load(fil)

  with open("/Users/shauryajain/alien-rematch/project3/AI-Project3-Code/model3-utils/alienautoenc.pkl","rb") as fil:
    alien_enc = pickle.load(fil)

  with open("/Users/shauryajain/alien-rematch/project3/AI-Project3-Code/model3-utils/crewautoenc.pkl","rb") as fil:
    crew_enc = pickle.load(fil)


  data_transformed = pt_alien.transform(X_train_inp[:,2500:5000])
  data_scaled = scaler_alien.transform(data_transformed)
  data_reduced = alien_enc.predict(data_scaled,verbose = 0)
  df_belief_network_alien = pd.DataFrame(data_reduced)
  df_belief_network_alien = df_belief_network_alien.fillna(-999)
  for i in df_belief_network_alien:
      if df_belief_network_alien[i].mean() < 0:
          df_belief_network_alien = df_belief_network_alien.drop(columns = i)
  z_scores = np.abs(stats.zscore(df_belief_network_alien))
  df_belief_network_alien = z_scores
  df_belief_network_alien = df_belief_network_alien.fillna(-999)
  for i in df_belief_network_alien:
      if df_belief_network_alien[i].mean() < 0:
          df_belief_network_alien = df_belief_network_alien.drop(columns = i)



  data_transformed = pt_crew.transform(X_train_inp[:,5000:7500])
  data_scaled = scaler_crew.transform(data_transformed)
  data_reduced = crew_enc.predict(data_scaled,verbose = 0)
  df_belief_network_crew = pd.DataFrame(data_reduced)
  df_belief_network_crew = df_belief_network_crew.fillna(-999)
  for i in df_belief_network_crew:
      if df_belief_network_crew[i].mean() < 0:
          df_belief_network_crew = df_belief_network_crew.drop(columns = i)
  z_scores = np.abs(stats.zscore(df_belief_network_crew))
  df_belief_network_crew = z_scores
  df_belief_network_crew = df_belief_network_crew.fillna(-999)
  for i in df_belief_network_crew:
      if df_belief_network_crew[i].mean() < 0:
          df_belief_network_crew = df_belief_network_crew.drop(columns = i)


  alien_red = np.array(df_belief_network_alien)
  crew_red = np.array(df_belief_network_crew)
  move_use = np.load("/Users/shauryajain/alien-rematch/project3/AI-Project3-Code/pos_moves_X_train.npy")[start_i:end_i]
  b_pos = X_train_inp[:,-4:-2]
  X_train_processed = np.concatenate((alien_red, crew_red, b_pos, move_use), axis=-1)

  return X_train_processed

from keras.models import Sequential
from keras.layers import Dense, Dropout

sa = 0
ea=3000

sta = -1500
eta = -500

sc = 0
ec = 3000

stc = -1500
etc = -500
critic_accuracy = []
actor_accuracy = []


for i in range(10):

  x_temp_train = np.load("/content/drive/MyDrive/AI-Project-3-data/PCS/Model 3 Data/X_train.npy")[sa:ea, :-2]
  X_train_actor = preprocess_dat(x_temp_train, 0, len(x_temp_train))

  #X_test for Actor
  X_temp_test = np.load("/content/drive/MyDrive/AI-Project-3-data/PCS/Model 3 Data/X_train.npy")[sta:eta, :-2]
  X_test_actor = preprocess_dat(X_temp_test, 0, len(X_temp_test))
  y_test = np.load("/content/drive/MyDrive/AI-Project-3-data/PCS/Model 3 Data/y_train.npy")[sta:eta]

  # Train data for Critic
  X_train_temp2 = np.load("/content/drive/MyDrive/AI-Project-3-data/PCS/Model 3 Data/X_train150.npy")
  y_train_temp2 = np.load("/content/drive/MyDrive/AI-Project-3-data/PCS/Model 3 Data/y_train150.npy")
  X_train_critic = X_train_temp2[sc:ec]
  X_test_critic = X_train_temp2[stc:etc]
  y_train_critic = y_train_temp2[sc:ec]
  y_test_critic = y_train_temp2[stc:etc]

  X_train_critic = preprocess_dat(X_train_critic, 0, len(X_train_critic))
  X_test_critic = preprocess_dat(X_test_critic, 0, len(X_test_critic))
  inp_size = 49

  critic_model = models.Sequential()
  critic_model.add(layers.Dense(16, activation='relu', input_shape=(inp_size,)))
  critic_model.add(Dropout(0.5))
  critic_model.add(layers.Dense(8, activation='relu'))
  critic_model.add(Dropout(0.5))
  critic_model.add(layers.Dense(4, activation='relu'))
  critic_model.add(Dropout(0.5))
  critic_model.add(layers.Dense(1, activation='sigmoid'))
  critic_model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

  critic_model.fit(X_train_critic[:, :inp_size], y_train_critic, epochs=50, batch_size=32, validation_split=0.1,verbose = 0)
  probabilities = critic_model.predict(X_test_critic[:,:inp_size],verbose = 0)
  etc = stc
  stc -= 1000

  pred = []
  for i in range(len(probabilities)):
    if probabilities[i]> 0.5:
      pred.append(1)
    else:
      pred.append(0)

  c_accuracy = accuracy_score(y_test_critic, np.array(pred))
  critic_accuracy.append(c_accuracy)
  print("Critic Accuracy: ",c_accuracy)


  del X_train_critic
  del X_test_critic
  sc += 1000
  ec += 1000

  preds = {}
  for i in range(0,4):
    move = np.full((len(x_temp_train), 1), i)
    X_input_critic = np.hstack((x_temp_train, move))
    X_input_critic = preprocess_dat(X_input_critic, 0, len(X_input_critic))


    preds[i] = list(critic_model.predict(X_input_critic[:,:inp_size],verbose = 0))

  y_train_actor = []
  for i in range(len(X_input_critic)):
    temp = [preds[0][i], preds[1][i], preds[2][i], preds[3][i]]

    ans = temp.index(max(temp))
    y_train_actor.append(ans)
  y_train_actor = np.array(y_train_actor)


  inp_size2 = 49
  y_train_actor[-1] = 1
  y_train_actor[-2] = 2
  y_train_actor[-3] = 3

  actor_model = Sequential()
  actor_model.add(Dense(32, input_dim=inp_size, activation='relu'))
  actor_model.add(Dropout(0.5))
  actor_model.add(Dense(16, activation='relu'))
  actor_model.add(Dropout(0.5))
  actor_model.add(Dense(8, activation='relu'))
  actor_model.add(Dropout(0.5))
  actor_model.add(Dense(4, activation='relu'))
  actor_model.add(Dropout(0.5))
  actor_model.add(Dense(4, activation='softmax'))
  actor_model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
  actor_model.fit(X_train_actor[:,:inp_size2], to_categorical(y_train_actor), epochs=50, batch_size=32, validation_split=0.1,verbose = 0)
  train_predictions = actor_model.predict(X_train_actor[:,:inp_size2],verbose = 0)
  sa = ea
  ea += 5000


  train_accuracy = accuracy_score(y_train_actor, np.argmax(train_predictions, axis=-1))
  print(f"Actor Accuracy (Train): {train_accuracy:.2f}")
  y_pred = actor_model.predict(X_test_actor[:,:inp_size2],verbose = 0)
  a_accuracy = accuracy_score(np.argmax(y_test, axis=-1),np.argmax(y_pred, axis=-1))
  actor_accuracy.append(a_accuracy)
  print(f"Actor Accuracy (Test): {a_accuracy:.2f}")
  eta = sta
  sta -= 1000
