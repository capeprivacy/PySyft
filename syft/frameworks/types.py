from typing import Union

from syft import dependency_check

framework_tensors = []
framework_shapes = []

if dependency_check.tensorflow_available:
    import tensorflow as tf
    from tensorflow.python.framework.ops import EagerTensor

    framework_tensors.append(EagerTensor)
    framework_shapes.append(tf.TensorShape)

if dependency_check.torch_available:
    import torch

    framework_tensors.append(torch.Tensor)
    framework_shapes.append(torch.Size)

try:
    framework_tensors = tuple(framework_tensors)
    FrameworkTensorType = Union[framework_tensors]
    FrameworkTensor = framework_tensors

    framework_shapes = tuple(framework_shapes)
    FrameworkShapeType = Union[framework_shapes]
    FrameworkShape = framework_shapes
except TypeError:  # no framework is installed
    FrameworkTensorType = None
    FrameworkShapeType = None
    FrameworkTensor = None
    FrameworkShape = None
