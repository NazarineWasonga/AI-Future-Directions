import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path='../models/edge_model.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load sample image
img = image.load_img('../data/sample_images/test/sample1.jpg', target_size=(64,64))
img_array = image.img_to_array(img)/255.0
img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

# Run inference
interpreter.set_tensor(input_details[0]['index'], img_array)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])
predicted_class = np.argmax(output)
print("Predicted class index:", predicted_class)
