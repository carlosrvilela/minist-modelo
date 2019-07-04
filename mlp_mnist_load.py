import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist


(a, b), (x_test, y_test) = mnist.load_data()
x_test = x_test[:1000,:,:]
y_test  = y_test[:1000]
x_test = x_test.reshape(1000, 784)
x_test = x_test.astype('float32')
x_test /= 255
y_test = tf.keras.utils.to_categorical(y_test, 10)

saved_model = load_model('final_model_966.h5')
saved_model.compile(loss='categorical_crossentropy', optimizer=tf.train.AdamOptimizer(), metrics=['accuracy'])
score = saved_model.evaluate(x_test, y_test, verbose=1)

print('Test loss:', score[0])
print('Test accuracy:', score[1])


