import logging
import logging.handlers
import multiprocessing
import multiprocessing.queues as mpq
import threading
import os

from time import time, sleep
from distutils.dir_util import copy_tree
from dataclasses import dataclass

import json

from namedqueues import NamedMultiProcessingQueue
from gamegraph import GameGraph
from processes.trainer import Trainer, TrainConfig
from processes.predictor import Predictor, PredictConfig
from processes.selfplayer import Selfplayer, SelfPlayConfig

LOGGER = logging.getLogger(__name__)

def getLogger():
    return LOGGER

@dataclass
class GymConfig():
    predict: PredictConfig
    selfplay: SelfPlayConfig 
    train: TrainConfig
    logsizelimit_kB: float = 100
    logbackups: int = 9
    loglevel: str = "DEBUG"
    # TODO: add loading from file

@dataclass
class GymPath():
    basefolder : str

    @property
    def config_file(self) ->str:
        return "{}/config.json".format(self.basefolder)        
    @property
    def weights_folder(self) ->str:
        return "{}/weights/".format(self.basefolder)
    @property
    def currentmodel_folder(self) ->str:
        return "{}/currentmodel/".format(self.basefolder)
    @property
    def gamerecordpool_folder(self) ->str:
        return "{}/gamerecordpool/".format(self.basefolder)
    @property
    def log_folder(self) ->str:
        return "{}/logs/".format(self.basefolder)
    @property
    def subfolders(self) -> list:
        return [self.currentmodel_folder, self.gamerecordpool_folder, 
        self.log_folder, self.weights_folder]


class GameGym():
    def __init__(self,         
        session_path: str, 
        graph: GameGraph,
        intialmodelpath: str = None,
        gym_config: GymConfig = None):
        self.graph = graph
        self._path = GymPath(session_path)

        if os.path.exists(self.path.basefolder):
            for p in self.path.subfolders:
                if not os.path.exists(p):
                    raise Exception("Can not load GameGym from basefolder '{}' - missing folder '{}'".format(self.path.basefolder, p))
        else:           
            for p in self.path.subfolders:                
                    os.makedirs(p)
        
        if gym_config == None:
            raise Exception("Can not initialize GameGym, loading config not yet implemented")
        else:
            self.config = gym_config
        
        logpath = os.path.join(self.path.log_folder,"{}.log".format(type(self).__name__))

        rfh = logging.handlers.RotatingFileHandler(logpath, 
                maxBytes=self.config.logsizelimit_kB*1000, backupCount=self.config.logbackups)
        rfh.setLevel(self.config.loglevel)
        rfh.setFormatter( logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s'))
        LOGGER.addHandler(rfh)
        if not intialmodelpath is None:            
            copy_tree(intialmodelpath, self.path.currentmodel_folder)
            LOGGER.debug("copied initial model from '{}' to '{}'".format(intialmodelpath, self.path.currentmodel_folder))
       
  
    @property
    def path(self):
        return self._path

    def resume(self):
        # ----------------------------------------------------------------------
        # 1. Start - initialize ressources
        # ----------------------------------------------------------------------       
        
        # events         
        endevent = multiprocessing.Event()
        newmodelevent = multiprocessing.Event()
        events = [endevent, newmodelevent]
        
        # queues
        logging_q = NamedMultiProcessingQueue("logging_q")
        predict_request_q = NamedMultiProcessingQueue("predict_request_q")
        predict_response_q = NamedMultiProcessingQueue("predict_response_q")
        queues = [logging_q, predict_request_q, predict_response_q]
        
        # start threads
        loggingthread = threading.Thread(
            target=self.__class__._distributed_logger,
            args=(logging_q,), 
            name="logging_thread")
        keyboard_listener_thread= threading.Thread(
            target=self._keyboard_listener,
            args=(endevent,),
            name="keyboard_reader")     
        threads = [loggingthread, keyboard_listener_thread]
        
        # processes   
        trainer_proc = Trainer(
                self.config.train,
                endevent, 
                newmodelevent,
                logging_q, 
                self.path.currentmodel_folder, 
                self.path.weights_folder, 
                self.path.gamerecordpool_folder)

        predictor_proc = Predictor(
                self.config.predict,
                endevent,
                newmodelevent,
                logging_q,
                predict_request_q,
                predict_response_q,
                self.path.currentmodel_folder)

        selfplayer_proc = Selfplayer(self.config.selfplay,
                endevent,
                logging_q,
                predict_request_q,
                predict_response_q,
                self.graph,
                self.path.gamerecordpool_folder)          
        
        processes = [trainer_proc, predictor_proc, selfplayer_proc]

        # ----------------------------------------------------------------------
        # 2. Run - start threads, process and handle until endevent
        # ----------------------------------------------------------------------
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

        # ----------------------------------------------------------------------
        # 3. Endphase - cleaning up ressources
        # ----------------------------------------------------------------------
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
    def _keyboard_listener(endevent : threading.Event):
        """Entry for a thread waiting for keyboard input."""
        threadname = threading.currentThread().getName()
        logger = logging.getLogger(threadname)
        logger.setLevel(logging.DEBUG)
        logger.info('keyboard_listener started')
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

            