import tensorflow as tf

# Load a pretrained model
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Save the model
model.save('mobilenetv2_model.h5')  # model.save('mobilenetv2_model.keras')
