from tflite_runtime.interpreter import Interpreter 
from PIL import Image
import numpy as np

def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image

def classify_image(interpreter, image, top_k=1):
    set_input_tensor(interpreter, image)
    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    output = np.squeeze(interpreter.get_tensor(output_details['index']))
    scale, zero_point = output_details['quantization']
    output = scale * (output - zero_point)
    ordered = np.argpartition(-output, 1)
    return [(i, output[i]) for i in ordered[:top_k]][0]

data_folder = "/home/pi/TFLite_MobileNet/"
model_path = data_folder + "model1.tflite"

interpreter = Interpreter(model_path)
print("Model Loaded Successfully.")

interpreter.allocate_tensors()
_, height, width, _ = interpreter.get_input_details()[0]['shape']
print("Image Shape (", width, ",", height, ")")

image = Image.open(data_folder + "test.jpg").convert('RGB').resize((width, height))
label_id, prob = classify_image(interpreter, image)

if(label_id == 0):
    print("Healthy", label_id)
else:
    print("Not healthy", label_id)
