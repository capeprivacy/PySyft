import logging

import syft
from syft import workers
from syft.workers.base import BaseWorker
from syft.frameworks.hook import BaseHook


class TensorFlowHook(BaseHook):
    def __init__(self, tensorflow, local_worker: BaseWorker = None, is_client: bool = True):
        self.torch = tensorflow
        self.tensorflow = tensorflow
        syft.torch = tensorflow
        syft.tensorflow = tensorflow

        self.local_worker = local_worker

        if hasattr(tensorflow, "tf_hooked"):
            logging.warning("TF was already hooked... skipping hooking process")
            self.local_worker = syft.local_worker
            return
        else:
            tensorflow.tf_hooked = True

        if self.local_worker is None:
            # Every TorchHook instance should have a local worker which is
            # responsible for interfacing with other workers. The worker
            # interface is what allows the Torch specific code in TorchHook to
            # be agnostic to the means by which workers communicate (such as
            # peer-to-peer, sockets, through local ports, or all within the
            # same process)
            self.local_worker = workers.VirtualWorker(
                hook=self, is_client_worker=is_client, id="me"
            )
        else:
            self.local_worker.hook = self

        self.to_auto_overload = {}

        self.args_hook_for_overloaded_attr = {}

        self._hook_native_tensor(torch.Tensor, TorchTensor)

    
    def _hook_native_tensor(self, tensor_type: type, syft_type: type):
        """Adds PySyft Tensor Functionality to the given native tensor type.

        Overloads the given native Torch tensor to add PySyft Tensor
        Functionality. Overloading involves modifying the tensor type with
        PySyft's added functionality. You may read about what kind of
        modifications are made in the methods that this method calls.

        Args:
            tensor_type: The type of tensor being hooked (in this refactor
                this is only ever torch.Tensor, but in previous versions of
                PySyft this iterated over all tensor types.
            syft_type: The abstract type whose methods should all be added to
                the tensor_type class. In practice this is always TorchTensor.
                Read more about it there.
        """
        # Reinitialize init method of Torch tensor with Syft init
        self._add_registration_to___init__(tensor_type, torch_tensor=True)

        # Overload Torch tensor properties with Syft properties
        self._hook_properties(tensor_type)

        # Returns a list of methods to be overloaded, stored in the dict to_auto_overload
        # with tensor_type as a key
        self.to_auto_overload[tensor_type] = self._which_methods_should_we_auto_overload(
            tensor_type
        )

        # [We don't rename native methods as torch tensors are not hooked] Rename native functions
        # #self._rename_native_functions(tensor_type)

        # Overload auto overloaded with Torch methods
        self._add_methods_from__torch_tensor(tensor_type, syft_type)

        self._hook_native_methods(tensor_type)

    
    def _add_registration_to___init__(hook_self, tensor_type: type, torch_tensor: bool = False):
        """Adds several attributes to the tensor.

        Overloads tensor_type.__init__ to add several attributes to the tensor
        as well as (optionally) registering the tensor automatically.
        TODO: auto-registration is disabled at the moment, this might be bad.

        Args:
            tensor_type: The type of tensor being hooked (in this refactor this
                is only ever torch.Tensor, but in previous versions of PySyft
                this iterated over all tensor types.
            torch_tensor: An optional boolean parameter (default False) to
                specify whether to skip running the native initialization
                logic. TODO: this flag might never get used.
        """
        if "native___init__" not in dir(tensor_type):
            tensor_type.native___init__ = tensor_type.__init__

        def new___init__(cls, *args, owner=None, id=None, register=True, **kwargs):
            initialize_tensor(
                hook_self=hook_self,
                cls=cls,
                id=id,
                is_tensor=torch_tensor,
                init_args=args,
                init_kwargs=kwargs,
            )

        tensor_type.__init__ = new___init__


    def _hook_properties(hook_self, tensor_type: type):
        """Overloads tensor_type properties.

        This method gets called only on torch.Tensor. If you're not sure how
        properties work, read:
        https://www.programiz.com/python-programming/property

        Args:
            tensor_type: The tensor type which is having properties
                added to it, typically just torch.Tensor.
        """

        @property
        def location(self):
            return self.child.location

        tensor_type.location = location

        @property
        def id_at_location(self):
            return self.child.id_at_location

        tensor_type.id_at_location = id_at_location

        @property
        def id(self):
            if not hasattr(self, "_id"):
                self._id = syft.ID_PROVIDER.pop()
            return self._id

        @id.setter
        def id(self, new_id):
            self._id = new_id
            return self

        tensor_type.id = id

        @property
        def owner(self):
            if not hasattr(self, "_owner"):
                self._owner = hook_self.local_worker
            return self._owner

        @owner.setter
        def owner(self, new_owner):
            self._owner = new_owner
            return self

        tensor_type.owner = owner

        @property
        def is_wrapper(self):
            if not hasattr(self, "_is_wrapper"):
                self._is_wrapper = False
            return self._is_wrapper

        @is_wrapper.setter
        def is_wrapper(self, it_is_a_wrapper):
            self._is_wrapper = it_is_a_wrapper
            return self

        tensor_type.is_wrapper = is_wrapper

        tensor_type.native_shape = tensor_type.shape
        tensor_type.native_data = tensor_type.data

        tensor_type.native_grad_fn = tensor_type.grad_fn

        def dim(self):
            return len(self.shape)

        tensor_type.dim = dim

        @property
        def grad_fn(self):
            if self.has_child():
                return self.child.grad_fn
            else:
                return self.native_grad_fn

        tensor_type.grad_fn = grad_fn


    def _which_methods_should_we_auto_overload(self, tensor_type: type):
        """Creates a list of Torch methods to auto overload.

        By default, it looks for the intersection between the methods of
        tensor_type and torch_type minus those in the exception list
        (syft.torch.exclude).

        Args:
            tensor_type: Iterate through the properties of this tensor type.
            syft_type: Iterate through all attributes in this type.

        Returns:
            A list of methods to be overloaded.
        """

        boolean_comparators = ["__gt__", "__ge__", "__lt__", "__le__"]

        to_overload = boolean_comparators

        for attr in dir(tensor_type):

            # Conditions for overloading the method
            if attr in syft.torch.exclude:
                continue
            if not hasattr(tensor_type, attr):
                continue

            lit = getattr(tensor_type, attr)
            is_base = attr in dir(object)
            is_desc = inspect.ismethoddescriptor(lit)
            is_func = isinstance(lit, types.FunctionType)
            try:
                is_service_func = "HookService" in lit.__qualname__
            except AttributeError:
                is_service_func = False
            is_overloaded = re.match("native*", attr) is not None

            if (is_desc or (is_func and not is_service_func)) and not is_base and not is_overloaded:
                to_overload.append(attr)

        return set(to_overload)

    @staticmethod
    def _add_methods_from__torch_tensor(tensor_type: type, syft_type: type):
        """Adds methods from the TorchTensor class to the native torch tensor.

        The class TorchTensor is a proxy to avoid extending directly the torch
        tensor class.

        Args:
            tensor_type: The tensor type to which we are adding methods
                from TorchTensor class.
        """
        exclude = [
            "__class__",
            "__delattr__",
            "__dir__",
            "__doc__",
            "__dict__",
            "__format__",
            "__getattribute__",
            "__hash__",
            "__init__",
            "__init_subclass__",
            "__weakref__",
            "__ne__",
            "__new__",
            "__reduce__",
            "__reduce_ex__",
            "__setattr__",
            "__sizeof__",
            "__subclasshook__",
            "_get_type",
            # "__eq__", # FIXME it now overwritten in native.py to use torch.eq, because of pb between == & __eq__ See #2030
            "__gt__",
            "__ge__",
            "__lt__",
            "__le__",
        ]
        # For all methods defined in TorchTensor which are not internal methods (like __class__etc)
        for attr in dir(syft_type):
            if attr not in exclude:
                if hasattr(tensor_type, attr):
                    setattr(tensor_type, f"native_{attr}", getattr(tensor_type, attr))
                # Add to the native tensor this method
                setattr(tensor_type, attr, getattr(TorchTensor, attr))

    def _hook_native_methods(self, tensor_type: type):
            """
            Add hooked version of all methods of to_auto_overload[tensor_type]
            to the tensor_type; instead of performing the native tensor
            method, the hooked version will be called

            Args:
                tensor_type: the tensor_type which holds the methods
            """
            # # Add methods defined in the TorchTensor class to the Pointer class
            # self._add_methods_from__torch_tensor(PointerTensor, TorchTensor)

            # Use a pre-defined list to select the methods to overload
            for attr in self.to_auto_overload[tensor_type]:
                # if we haven't already overloaded this function
                if f"native_{attr}" not in dir(tensor_type):
                    native_method = getattr(tensor_type, attr)
                    setattr(tensor_type, f"native_{attr}", native_method)
                    new_method = self.get_hooked_method(attr)
                    setattr(tensor_type, attr, new_method)