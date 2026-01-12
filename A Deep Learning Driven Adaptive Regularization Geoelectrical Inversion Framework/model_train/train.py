
import tensorflow as tf
import numpy as np

from model_denoisy.DeCnn2 import *

CONFIG = {
    'batch_size': 256,
    'num_epochs': 200,
    'learning_rate': 1e-3,
    'min_lr': 1e-6,
}

def prepare_data_for_keras(noisy, clean):
    return np.stack(noisy, axis=0), np.stack(clean, axis=0)

print(f"Config: {CONFIG}")

# Model Setup
model = fcn_denoiser2()

# Data Loading
# Dl-regularization data
train_x_np = np.abs(np.expand_dims(np.load('./model_train\data_create/train_x.npy'), axis=-1))
train_y_np = np.abs(np.expand_dims(np.load('./model_train\data_create/train_y.npy'), axis=-1))

# # imaging enhancedment data 
# train_x_np = np.abs(np.expand_dims(np.load('./model_train\data_create/train_y.npy'), axis=-1))
# train_y_np = np.abs(np.expand_dims(np.load('./model_train\data_create/train_mask.npy'), axis=-1))

# Prepare Data
train_x_keras, train_y_keras = prepare_data_for_keras(train_x_np, train_y_np)

# Learning Rate Scheduler
class SimulatedAnnealingScheduler(tf.keras.callbacks.Callback):
    def __init__(self, initial_lr, min_lr, max_epochs):
        super().__init__()
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.max_epochs = max_epochs
    def on_epoch_begin(self, epoch, logs=None):
        lr = self.min_lr + (self.initial_lr - self.min_lr) * np.exp(-epoch / self.max_epochs)
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=30, restore_best_weights=True, verbose=1
)

simulated_annealing_scheduler = SimulatedAnnealingScheduler(
    CONFIG['learning_rate'], CONFIG['min_lr'], CONFIG['num_epochs']
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=CONFIG['learning_rate']),
    loss='mse',
)

history = model.fit(
    train_x_keras, train_y_keras,
    epochs=CONFIG['num_epochs'],
    batch_size=CONFIG['batch_size'],
    shuffle=True,
    validation_split=0.1,
    verbose=1,
    callbacks=[early_stopping, simulated_annealing_scheduler]
)

for i in range(len(model.weights)):
    model.weights[i]._handle_name = model.weights[i].name + "_:" + str(i)

model.save('D:\教学\深度学习正则化\拟2DTEM正则化\Train\数据集/reg_model.h5')
# model.save('D:\教学\深度学习正则化\拟2DTEM正则化\Train\数据集/imaging_model.h5')
print(f"Model weights saved")

