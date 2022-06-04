import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model('xception_v4_12_0.868.h5')

tf.saved_model.save(model, 'clothing-model')


# To see what the model contains you can run
# saved_model_cli show --dir clothing-model --all