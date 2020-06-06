import multiprocessing.queues
import queue

class NamedMultiProcessingQueue(multiprocessing.queues.Queue):
    """Adding a name atttibute to multiprocessing.queues.Queue."""
    def __init__(self, name, maxsize=0):
        super().__init__(maxsize, ctx=multiprocessing.get_context())
        self.name = name

class NamedQueue(queue.Queue):
    """Adding a name atttibute to queue.Queue."""
    def __init__(self, name, maxsize=0):
        super().__init__(maxsize)
        self.name = name