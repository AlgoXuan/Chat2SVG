import torch
import pydiffvg
import diffusers
import transformers
import picosvg
import cssutils
import skimage
import numpy

print('-' * 30)
print(f'Numpy Version: {numpy.__version__} (Should be < 2.0)')
print(f'Torch Version: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
print(f'Picosvg Location: {picosvg.__file__}')
print('All critical modules imported successfully!')
print('-' * 30)