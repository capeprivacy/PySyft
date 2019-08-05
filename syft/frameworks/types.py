# from __future__ import absolute_import
from typing import Union

from syft import dependency_check

framework_tensors = []
hooks = []

if dependency_check.tensorflow_available:
    from tensorflow.python.framework.ops import EagerTensor

    framework_tensors.append(EagerTensor)

if dependency_check.torch_available:
    import torch

    framework_tensors.append(torch.Tensor)

FrameworkTensor = None
for tensor_type in framework_tensors:
    if FrameworkTensor is None:
        FrameworkTensor = tensor_type
    else:
        FrameworkTensor = Union[FrameworkTensor, tensor_type]
