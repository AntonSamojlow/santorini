import logging
import logging.handlers
import multiprocessing
import multiprocessing.queues as mpq
import threading

from os import path, makedirs
from time import time, sleep

from namedqueues import NamedMultiProcessingQueue
from gamegraph import GameGraph
from processes.trainer import Trainer, TrainConfig
from processes.predictor import Predictor 
from processes.selfplayer import Selfplayer, SelfPlayConfig

LOGGER = logging.getLogger(__name__)

def getLogger():
    return LOGGER

class GymSettings():
    def __init__(self, 
        mcts_config : SelfPlayConfig, 
        train_config : TrainConfig):
        self.mcts_config = mcts_config
        self.train_config = train_config


class GameGym():
    def __init__(self, graph, settings, sessionfolder, intialmodelpath = None):
        self.graph = graph
        if not isinstance(settings, GymSettings):
            LOGGER.error("settings need to be of type {}".format(GymSettings.__class__))
        self.settings = settings
        self._sessionfolder = sessionfolder
        if intialmodelpath is None:
            self._modelpath = "{}/models/current/".format(self.sessionfolder)
        else:
            self._modelpath = intialmodelpath

    @property
    def sessionfolder(self):
        return self.sessionfolder

    @property
    def modelpath(self):
        return self._modelpath

    def resume(self):
        # 1. Start - initialize ressources         
        # # events         
        endevent = multiprocessing.Event()
        newmodelevent = multiprocessing.Event()
        events = [endevent, newmodelevent]
        # queues
        logging_q = NamedMultiProcessingQueue("logging_q")
        predict_request_q = NamedMultiProcessingQueue("predict_request_q")
        predict_response_q = NamedMultiProcessingQueue("predict_response_q")
        queues = [logging_q, predict_request_q, predict_response_q]
        # start threads
        loggingthread = threading.Thread(target=self.__class__._distributed_logger,
            args=(logging_q,), name="logging_thread")
        keyboard_reader_thread= threading.Thread(
            target=self._keyboard_reader,
            args=(endevent, ),
            name="keyboard_reader")

     
        threads = [loggingthread, keyboard_reader_thread]
        # processes   
        trainer_proc = Trainer(self.modelpath, self.settings.train_config, 
                                            endevent, newmodelevent, logging_q)
        predictor_proc = Predictor(logging_q, predict_request_q, predict_response_q,
                    endevent, newmodelevent, 10, self.modelpath)      
        selfplayer_proc = Selfplayer(self.graph, self.settings.mcts_config, endevent, 
                    logging_q, predict_request_q, predict_response_q)
        
        processes = [trainer_proc, predictor_proc, selfplayer_proc]

        # 2. Run...
        # start threads
        for t in threads:
            t.start()
        while not loggingthread.is_alive():
            sleep(0.1)
        LOGGER.info("loggingthread alive")    

        # start processes
        for p in processes:
            p.start()
        
        while not endevent.is_set():
            sleep(1)

        # 3. Endphase - cleaning up ressources
        LOGGER.info("endevent was set")
        logging_q.put('TER')
        
        for p in processes:
            p.join(10)          
            if p.is_alive():
                LOGGER.warning("process {} has not ended - using terminate command".format(p.name))
                p.terminate()                
            else:
                LOGGER.info("process {} has ended".format(p.name))
        
        for t in threads:
            t.join(10)
            if t.is_alive():
                LOGGER.warning("thread {} has not ended".format(t.name))
            else:
                LOGGER.info("thread {} has ended".format(t.name))

        for q in queues:
            if not q.empty():
                LOGGER.info("{} items on queue {} - emptying".format(q.qsize(), q.name))
                while not q.empty():
                    q.get()
            if isinstance(q, mpq.Queue):
                q.close()
                q.join_thread()            
                LOGGER.info("queue {} joined".format(q.name))
        LOGGER.info("ending")
        return
    
    @staticmethod
    def _keyboard_reader(endevent : threading.Event):
        """Entry for a thread waiting for keyboard input."""
        threadname = threading.currentThread().getName()
        logger = logging.getLogger(threadname)
        logger.setLevel(logging.DEBUG)
        logger.info('keyboard_watcher started')
        while True:
            text = input()
            logger.info("registered input '{}'".format(text))
            if text == "exit":
                endevent.set()
                return
           
    @staticmethod
    def _distributed_logger(logging_q : mpq.Queue):
        """Entry for a thread handling logging from the workers via queues."""
        threadname = threading.currentThread().getName()
        logger = logging.getLogger(threadname)
        logger.info('distributed_logger starting')
        while True:
            record = logging_q.get(block=True)
            if record == 'TER':
                logger.info('received {}'.format(record))
                break
            rootlogger = logging.getLogger()
            rootlogger.handle(record)
  
