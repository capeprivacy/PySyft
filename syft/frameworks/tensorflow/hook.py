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
