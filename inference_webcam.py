import tensorrt as trt
import cv2
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

from utils_cuda import HostDeviceMem

from utils_inference import letterbox

IMG_SIZE = 640
WEIGHT = './yolov5n.trt'

logger = trt.Logger(trt.Logger.INFO)

with open(WEIGHT, 'rb') as f, trt.Runtime(logger) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

# Dans cette partie on regarde les allocations memoires

input = []
outputs = []
bindings = []
stream = cuda.Stream()

for binding in engine:
    shape = engine.get_tensor_shape(binding)
    dtype = trt.nptype(engine.get_tensor_dtype(binding))
    
    host_mem = np.empty(shape, dtype=dtype)
    device_mem = cuda.mem_alloc(host_mem.nbytes) #allouer une place dans le GPU pour chaque binding
    bindings.append(int(device_mem)) #list des address dans le GPU

    if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
        shape = engine.get_tensor_shape(binding)
        print('input', binding)
        input.append(HostDeviceMem(host_mem, device_mem))
    else:
        outputs.append(HostDeviceMem(host_mem, device_mem))
        print('output', binding)

context = engine.create_execution_context()

cap = cv2.VideoCapture(0)

while True:
    _, im0 = cap.read()

    im = letterbox(im0, IMG_SIZE)[0]
    im = im.transpose((2, 0, 1))[::-1]  # BGR to RGB
    im = np.ascontiguousarray(im)
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    np.copyto(input[0].host, im)
    cuda.memcpy_htod_async(input[0].device, input[0].host)
    succes = context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(outputs[-1].host, outputs[-1].device)

    print('ok')

