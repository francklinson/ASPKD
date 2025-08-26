import torch
import tensorflow as tf

print("torch version: ", torch.__version__)
if torch.cuda.is_available():
    print("Cuda available!")
    print(torch.cuda.current_device())
    print(torch.cuda.device_count())
else:
    print("Cuda not available in this env!")

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")
print("tensorflow version: ", tf.__version__)

# pip freeze > requirements.txt
# pip install -r requirements.txt
# git ls-files | xargs wc -l