import syft

if syft.dependency_check.tensorflow_available:
    from syft_tensorflow.hook import TensorFlowHook
    from syft_tensorflow.tensor import TensorFlowTensor
    # Add Tensorflow type_rule to syft type rule
    from syft_tensorflow.hook.hook_args import type_rule
    setattr(syft, "TensorFlowHook", TensorFlowHook)
