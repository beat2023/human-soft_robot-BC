# Transform data into time series and apply LSTM inverse dynamic model

import pandas as pd
from io import StringIO
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.python.keras.models import load_model
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statistics import mean

########################################################################################################################
# PREPARE DATA

data_0 = open('../Babbling/DataFolder/origin_ref/shift_idm_sinusoids_0.txt', 'r').read()
data_1 = open('../Babbling/DataFolder/origin_ref/shift_idm_parabola_0.txt', 'r').read()
data_2 = open('../Babbling/DataFolder/origin_ref/shift_idm_random_0.txt', 'r').read()
data_3 = open('../Babbling/DataFolder/origin_ref/shift_idm_plateau_0.txt', 'r').read()
data_4 = open('../Babbling/DataFolder/origin_ref/shift_idm_circles_0.txt', 'r').read()

file = data_0 + "\n" + data_1 + "\n" + data_2 + "\n" + data_3 + "\n" + data_4

file_1 = file.replace(',\n', ",")
file_1 = file_1.replace('])', "")
file_1 = file_1.replace('array([', "")
file_1 = file_1.replace('[', "")
file_1 = file_1.replace(']', "")

data = StringIO(file_1)
df = pd.read_csv(data, sep=",", names=["count", "P1", "P2", "P3", "P4", "P5", "P6",
                                       "pmx", "pmy", "pmz", "dmx", "dmy", "dmz"])

# Data for three-dimensional scattered points
all_data = df.loc[:, "pmx":].to_numpy()
zdata = all_data[:, [2, 5]]
ydata = all_data[:, [1, 4]]
xdata = all_data[:, [0, 3]]
# 3D plotting
# ax = plt.axes(projection='3d')
# ax.scatter3D(xdata[:, 0], ydata[:, 0], zdata[:, 0], color='r', label='proximal', s=1)
# ax.scatter3D(xdata[:, 1], ydata[:, 1], zdata[:, 1], color='b', label='distal', s=1)
# ax.set_xlabel('x', fontsize=15)
# ax.set_ylabel('y', fontsize=15)
# ax.set_zlabel('z', fontsize=15)
# ax.set_title('Babbling data of the robot')
# plt.legend()
# plt.show()

# Position for proximal and distal segments (x, y, and z)
positions = df.loc[:, "pmx":].to_numpy()
# Pressure values
pressures = df.loc[:, "P1":"P6"].to_numpy()

# Saving minima and maxima of positions and pressures
max_pos = np.amax(positions, axis=0)
min_pos = np.amin(positions, axis=0)
max_press = np.amax(pressures, axis=0)
min_press = np.amin(pressures, axis=0)

np.savetxt('Normalization_parameters/max_pos.txt', max_pos)
np.savetxt('Normalization_parameters/min_pos.txt', min_pos)
np.savetxt('Normalization_parameters/max_press.txt', max_press)
np.savetxt('Normalization_parameters/min_press.txt', min_press)

# Normalization in interval [-1, 1]
scaler_pos = MinMaxScaler(feature_range=(-1, 1))
scaler_pos.fit(positions)
pos_scaled = scaler_pos.transform(positions)

scaler_press = MinMaxScaler(feature_range=(-1, 1))
scaler_press.fit(pressures)
press_scaled = scaler_press.transform(pressures)

# Shifting positions and pressures to create features set
X_tot = pos_scaled
pos_prev = pos_scaled[:-1, :]
pos_prev = np.insert(pos_prev, 0, pos_scaled[0, :], axis=0)
pos_prev_2 = pos_prev[:-1, :]
pos_prev_2 = np.insert(pos_prev_2, 0, pos_prev[0, :], axis=0)

pos_next = pos_scaled[1:, :]
pos_next = np.append(pos_next, np.reshape(pos_scaled[-1, :], (1, 6)), axis=0)
pos_next_2 = pos_next[1:, :]
pos_next_2 = np.append(pos_next_2, np.reshape(pos_next[-1, :], (1, 6)), axis=0)
pos_next_3 = pos_next_2[1:, :]
pos_next_3 = np.append(pos_next_3, np.reshape(pos_next_2[-1, :], (1, 6)), axis=0)

press_prev = press_scaled[:-1, :]
press_prev = np.insert(press_prev, 0, press_scaled[0, :], axis=0)
press_prev_2 = press_prev[:-1, :]
press_prev_2 = np.insert(press_prev_2, 0, press_prev[0, :], axis=0)

# Concatenating all
X_tot = np.append(X_tot, pos_prev_2, axis=1)
X_tot = np.append(X_tot, pos_prev, axis=1)
X_tot = np.append(X_tot, pos_next, axis=1)
X_tot = np.append(X_tot, pos_next_2, axis=1)
X_tot = np.append(X_tot, pos_next_3, axis=1)
X_tot = np.append(X_tot, press_prev_2, axis=1)
X_tot = np.append(X_tot, press_prev, axis=1)

# Save X_tot and targets in a file
file_wname = 'BabblingData/all_normalized.txt'
with open(file_wname, 'w') as f:
    for d in range(X_tot.shape[0]):
        f.write(str([d, press_scaled[d, :].tolist(), X_tot[d, :].tolist()]).strip('[]'))
        f.write('\n')

# X used for the open loop training and testing
ol_part = 0.9
ol_index = int(X_tot.shape[0] * ol_part)
X_tot_ol = X_tot[:ol_index, :]
pressures_ol = pressures[:ol_index, :]
press_scaled_ol = press_scaled[:ol_index, :]
# X used for the closed loop testing
X_tot_cl = X_tot[ol_index:, :]
pressures_cl = pressures[ol_index:, :]
press_scaled_cl = press_scaled[ol_index:, :]

# Reshaping the data
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X_tot_ol, press_scaled_ol, test_size=0.1, shuffle=True)
Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], 1, Xtrain.shape[1]))
Ytrain = np.reshape(Ytrain, (Ytrain.shape[0], 1, Ytrain.shape[1]))
Xtest = np.reshape(Xtest, (Xtest.shape[0], 1, Xtest.shape[1]))
Ytest = np.reshape(Ytest, (Ytest.shape[0], 1, Ytest.shape[1]))

Ytest_reshaped = np.reshape(Ytest, (Ytest.shape[0], Ytest.shape[2]))
target_test = scaler_press.inverse_transform(Ytest_reshaped)


########################################################################################################################
# LSTM MODEL

def scheduler(epoch, lr):
    if epoch <= 50:
        return lr
    elif 50 < epoch <= 100:
        return lr * 0.9
    else:
        return lr * 0.1


my_callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=10),
    # tf.keras.callbacks.ModelCheckpoint(filepath='model_B/model_{epoch:02d}_IDM.h5',
    #                                    save_best_only=True),
    # tf.keras.callbacks.TensorBoard(log_dir='model_B/logs'),
    tf.keras.callbacks.LearningRateScheduler(scheduler)
]

# List of hyperparameters
activ_funct = 'tanh'
# activ_funct provato con ['softsign', 'tanh']
# similar performance
recurrent_activ = None
# recurrent_activ provato con [None, 'softsign', 'tanh']
# softsign e tanh pessimo
dropout = 0.1
# dropout provato con [0.1, 0.2, 0.3]
# via via peggiora
learn_rate = 0.0006
# learn_rate provato con [0.01, 0.001, 0.0006, 0.0001]
# best with 0.0006
epochs = 150
# epochs provato con [100, 150, 200]
# via via meglio se uso bassa learning rate
batch_size = 16
# batch_size provato con [8, 16, 32]
# with 8 and 16 similar

performance = []
performance_avg = []
performance_cl = []
performance_cl_avg = []
n_episodes = 1

# TO COMMENT TO USE THE ALREADY TRAINED MODEL
# for i in range(n_episodes):
#     model = Sequential()
#     model.add(LSTM(256, input_shape=(None, Xtrain.shape[2]), activation=activ_funct,
#                    recurrent_activation=recurrent_activ, return_sequences=True))
#     model.add(Dense(128, activation=activ_funct))
#     model.add(Dropout(dropout))
#     model.add(Dense(Ytrain.shape[2], activ_funct))
#     opt = Adam(learning_rate=learn_rate)
#     model.compile(optimizer=opt, loss='mse', metrics="mean_absolute_error")
#     model.summary()
#
#     # Fit the model
#     history = model.fit(Xtrain, Ytrain, epochs=epochs, batch_size=batch_size,
#                         validation_split=0.1, shuffle=False, verbose=1, callbacks=my_callbacks)
#
#     plt.plot(history.history['loss'], label='Training loss')
#     plt.plot(history.history['val_loss'], label='Validation loss')
#     plt.legend()
#     plt.show()
#
#     # Forecast and rescaling to original range
#     forecast = model.predict(Xtest)
#     forecast = np.reshape(forecast, (forecast.shape[0], forecast.shape[2]))
#     output_test = scaler_press.inverse_transform(forecast)
#
#     r2_test = r2_score(target_test, output_test, multioutput='raw_values')
#     r2_test_avg = mean(r2_test)
#     performance.append(r2_test)
#     performance_avg.append(r2_test_avg)
#     if r2_test_avg <= min(performance_avg):
#         best_model = model
#         best_output_test = output_test
#
#     # CLOSED LOOP PREDICTION
#     Xtest_cl = X_tot_cl[:, :-12]  # I consider only positions
#     Ytest_cl = pressures_cl[:, :]
#     press_past_2 = scaler_press.transform(np.zeros((1, 6)))  # initializing tau(t-2)
#     press_past = scaler_press.transform(np.zeros((1, 6)))  # initializing tau(t-1)
#     forecast_cl = np.zeros(Ytest_cl.shape)
#     output_test_cl = np.zeros(Ytest_cl.shape)
#     for j in range(Xtest_cl.shape[0]):
#         X_curr = np.append(Xtest_cl[j, :], press_past_2)
#         X_curr = np.append(X_curr, press_past)
#         forecast_curr = model.predict(np.reshape(X_curr, (1, 1, X_curr.shape[0])), verbose=1)
#         forecast_curr = np.reshape(forecast_curr, (forecast_curr.shape[0], forecast_curr.shape[2]))
#         press_past_2 = press_past  # Update the features in close loop
#         press_past = forecast_curr  # Update the features in close loop
#         forecast_cl[j, :] = forecast_curr
#     output_test_cl = scaler_press.inverse_transform(forecast_cl)
#     r2_test_cl = r2_score(Ytest_cl, output_test_cl, multioutput='raw_values')
#     r2_test_avg_cl = mean(r2_test_cl)
#
#     performance_cl.append(r2_test_cl)
#     performance_cl_avg.append(r2_test_avg_cl)

# TO COMMENT TO TRAIN ANOTHER TIME
best_model = load_model('IDM_MODEL')
forecast = best_model.predict(Xtest)
forecast = np.reshape(forecast, (forecast.shape[0], forecast.shape[2]))
best_output_test = scaler_press.inverse_transform(forecast)

########################################################################################################################
# PERFORMANCE
print('OPEN LOOP PERFORMANCE')

# Mean squared error
mse_test = mean_squared_error(target_test, best_output_test, multioutput='raw_values')
mse_test_avg = mean(mse_test)
print(f'MSE value for test set: {mse_test_avg}')
# Mean absolute error
mae_test = mean_absolute_error(target_test, best_output_test, multioutput='raw_values')
mae_test_avg = mean(mae_test)
print(f'MAE value for test set: {mae_test_avg}')
# R2 score
r2_test = r2_score(target_test, best_output_test, multioutput='raw_values')
r2_test_avg = mean(r2_test)
print(f'R2 score value for test set: {r2_test_avg}')

fig, axs = plt.subplots(3, 2)
plt.suptitle("Open-loop performance on open-loop test set")
axs[0, 0].set_title('Chambers proximal segment')
axs[0, 0].plot(best_output_test[:, 0], 'r', label="Predicted")
axs[0, 0].plot(target_test[:, 0], 'b', label="True")
axs[1, 0].plot(best_output_test[:, 1], 'r', label="Predicted")
axs[1, 0].plot(target_test[:, 1], 'b', label="True")
axs[2, 0].plot(best_output_test[:, 2], 'r', label="Predicted")
axs[2, 0].plot(target_test[:, 2], 'b', label="True")
axs[0, 1].set_title('Chambers distal segment')
axs[0, 1].plot(best_output_test[:, 3], 'r', label="Predicted")
axs[0, 1].plot(target_test[:, 3], 'b', label="True")
axs[1, 1].plot(best_output_test[:, 4], 'r', label="Predicted")
axs[1, 1].plot(target_test[:, 4], 'b', label="True")
axs[2, 1].plot(best_output_test[:, 5], 'r', label="Predicted")
axs[2, 1].plot(target_test[:, 5], 'b', label="True")
for ax in axs.flat:
    ax.set(xlabel='Sample number', ylabel='Pressure value')
# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()
plt.legend()
plt.show()

fig_0, axs_0 = plt.subplots(3, 2)
plt.suptitle("Open-loop performance: first 100 samples")
axs_0[0, 0].set_title('Chambers proximal segment')
axs_0[0, 0].plot(best_output_test[:100, 0], 'r', label="Predicted")
axs_0[0, 0].plot(target_test[:100, 0], 'b', label="True")
axs_0[1, 0].plot(best_output_test[:100, 1], 'r', label="Predicted")
axs_0[1, 0].plot(target_test[:100, 1], 'b', label="True")
axs_0[2, 0].plot(best_output_test[:100, 2], 'r', label="Predicted")
axs_0[2, 0].plot(target_test[:100, 2], 'b', label="True")
axs_0[0, 1].set_title('Chambers distal segment')
axs_0[0, 1].plot(best_output_test[:100, 3], 'r', label="Predicted")
axs_0[0, 1].plot(target_test[:100, 3], 'b', label="True")
axs_0[1, 1].plot(best_output_test[:100, 4], 'r', label="Predicted")
axs_0[1, 1].plot(target_test[:100, 4], 'b', label="True")
axs_0[2, 1].plot(best_output_test[:100, 5], 'r', label="Predicted")
axs_0[2, 1].plot(target_test[:100, 5], 'b', label="True")
for ax in axs_0.flat:
    ax.set(xlabel='Sample number', ylabel='Pressure value')
# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs_0.flat:
    ax.label_outer()
plt.legend()
plt.show()


########################################################################################################################
# CLOSED LOOP PREDICTION

# I update the values of pressure in closed loop
Xtest_cl = X_tot_cl[:, :-12]  # I consider only positions
Ytest_cl = pressures_cl[:, :]
press_past_2 = scaler_press.transform(np.zeros((1, 6)))  # initializing tau(t-2)
press_past = scaler_press.transform(np.zeros((1, 6)))  # initializing tau(t-1)
forecast_cl = np.zeros(Ytest_cl.shape)
output_test_cl = np.zeros(Ytest_cl.shape)
for i in range(Xtest_cl.shape[0]):
    X_curr = np.append(Xtest_cl[i, :], press_past_2)
    X_curr = np.append(X_curr, press_past)
    forecast_curr = best_model.predict(np.reshape(X_curr, (1, 1, X_curr.shape[0])), verbose=1)
    forecast_curr = np.reshape(forecast_curr, (forecast_curr.shape[0], forecast_curr.shape[2]))
    press_past_2 = press_past  # Update the features in close loop
    press_past = forecast_curr  # Update the features in close loop
    forecast_cl[i, :] = forecast_curr

# Rescaling to original range and rounding
output_test_cl = scaler_press.inverse_transform(forecast_cl)

########################################################################################################################
# PERFORMANCE
print('CLOSED LOOP PERFORMANCE')

# Mean squared error
mse_test_cl = mean_squared_error(Ytest_cl, output_test_cl, multioutput='raw_values')
mse_test_avg_cl = mean(mse_test_cl)
print(f'MSE value for test set: {mse_test_avg_cl}')
# Mean absolute error
mae_test_cl = mean_absolute_error(Ytest_cl, output_test_cl, multioutput='raw_values')
mae_test_avg_cl = mean(mae_test_cl)
print(f'MAE value for test set: {mae_test_avg_cl}')
# R2 score
r2_test_cl = r2_score(Ytest_cl, output_test_cl, multioutput='raw_values')
r2_test_avg_cl = mean(r2_test_cl)
print(f'R2 score value for test set: {r2_test_avg_cl}')

fig_1, axs_1 = plt.subplots(3, 2)
plt.suptitle("Closed-loop performance on closed-loop test set")
axs_1[0, 0].set_title('Chambers proximal segment')
axs_1[0, 0].plot(output_test_cl[:, 0], 'r', label="Predicted")
axs_1[0, 0].plot(Ytest_cl[:, 0], 'b', label="True")
axs_1[1, 0].plot(output_test_cl[:, 1], 'r', label="Predicted")
axs_1[1, 0].plot(Ytest_cl[:, 1], 'b', label="True")
axs_1[2, 0].plot(output_test_cl[:, 2], 'r', label="Predicted")
axs_1[2, 0].plot(Ytest_cl[:, 2], 'b', label="True")
axs_1[0, 1].set_title('Chambers distal segment')
axs_1[0, 1].plot(output_test_cl[:, 3], 'r', label="Predicted")
axs_1[0, 1].plot(Ytest_cl[:, 3], 'b', label="True")
axs_1[1, 1].plot(output_test_cl[:, 4], 'r', label="Predicted")
axs_1[1, 1].plot(Ytest_cl[:, 4], 'b', label="True")
axs_1[2, 1].plot(output_test_cl[:, 5], 'r', label="Predicted")
axs_1[2, 1].plot(Ytest_cl[:, 5], 'b', label="True")
for ax in axs_1.flat:
    ax.set(xlabel='Sample number', ylabel='Pressure value')
# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs_1.flat:
    ax.label_outer()
plt.legend()
plt.show()

fig_2, axs_2 = plt.subplots(3, 2)
plt.suptitle("Closed-loop performance: 200 samples")
axs_2[0, 0].set_title('Chambers proximal segment')
axs_2[0, 0].plot(output_test_cl[:200, 0], 'r', label="Predicted")
axs_2[0, 0].plot(Ytest_cl[:200, 0], 'b', label="True")
axs_2[1, 0].plot(output_test_cl[:200, 1], 'r', label="Predicted")
axs_2[1, 0].plot(Ytest_cl[:200, 1], 'b', label="True")
axs_2[2, 0].plot(output_test_cl[:200, 2], 'r', label="Predicted")
axs_2[2, 0].plot(Ytest_cl[:200, 2], 'b', label="True")
axs_2[0, 1].set_title('Chambers distal segment')
axs_2[0, 1].plot(output_test_cl[:200, 3], 'r', label="Predicted")
axs_2[0, 1].plot(Ytest_cl[:200, 3], 'b', label="True")
axs_2[1, 1].plot(output_test_cl[:200, 4], 'r', label="Predicted")
axs_2[1, 1].plot(Ytest_cl[:200, 4], 'b', label="True")
axs_2[2, 1].plot(output_test_cl[:200, 5], 'r', label="Predicted")
axs_2[2, 1].plot(Ytest_cl[:200, 5], 'b', label="True")
for ax in axs_2.flat:
    ax.set(xlabel='Sample number', ylabel='Pressure value')
# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs_2.flat:
    ax.label_outer()
plt.legend()
plt.show()

########################################################################################################################
# SAVING THE MODEL
# best_model.save('IDM_MODEL')  # 'IDM_MODEL' directory
