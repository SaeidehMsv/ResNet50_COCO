import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from data import make_tf
from model import create_model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

test_dataset, test_steps = make_tf('test_dataset.csv', 64)
train_dataset, train_steps = make_tf('train_dataset.csv', 64)
model = create_model()
# model.summary()
model.compile(loss=tf.keras.losses.mse, metrics=['mse'],
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
checkpoint_filepath = '/tmp/checkpoint'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_loss',
    mode='min',
    save_best_only=True)
history = model.fit(train_dataset,
                    steps_per_epoch=train_steps,
                    epochs=50, verbose=1,
                    validation_data=test_dataset,
                    validation_steps=test_steps,
                    callbacks=[model_checkpoint_callback])
