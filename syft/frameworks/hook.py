from abc import ABC

from syft.workers import BaseWorker


class BaseHook(ABC):
    def __init__(self, framework_module, local_worker: BaseWorker = None, is_client: bool = True):
        pass
