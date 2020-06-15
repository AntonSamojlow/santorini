import logging
import logging.handlers
import multiprocessing
import multiprocessing.queues as mpq
import threading
import os
from time import time, sleep
from distutils.dir_util import copy_tree

import json

import gymdata
from namedqueues import NamedMultiProcessingQueue
from gamegraph import GameGraph
from processes.trainer import Trainer
from processes.predictor import Predictor
from processes.selfplayer import Selfplayer

LOGGER = logging.getLogger(__name__)

def getLogger():
    return LOGGER

class GameGym(): 
    def __init__(self,         
        session_path: str, 
        graph: GameGraph,
        intialmodelpath: str = None,
        gym_config: gymdata.GymConfig = None):
        self.graph = graph
        self._path = gymdata.GymPath(session_path)

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
        
        logfilepath = os.path.join(self.path.log_folder,"{}.log".format(type(self).__name__))
        LOGGER.addHandler(gym_config.logging.getRotatingFileHandler(logfilepath))

        if not intialmodelpath is None:            
            copy_tree(intialmodelpath, self.path.currentmodel_folder)
            LOGGER.debug("copied initial model from '{}' to '{}'".format(intialmodelpath, self.path.currentmodel_folder))
            with open(self.path.currentmodelinfo_file, 'w+') as f:
                f.write(gymdata.ModelInfo(0).as_json(indent=0))
                LOGGER.debug("created new model info file (iteration count = 0)")
            
                        
  
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
                self.path,
                endevent, 
                newmodelevent,
                logging_q)

        predictor_proc = Predictor(
                self.config.predict,
                self.path,
                endevent,
                newmodelevent,
                logging_q,
                predict_request_q,
                predict_response_q)

        selfplayer_proc = Selfplayer(
                self.config.selfplay,
                self.path,
                endevent,
                logging_q,
                predict_request_q,
                predict_response_q,
                self.graph)          
        
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
        """Entry for a thread handling logging from subprocesses via queues."""
        threadname = threading.currentThread().getName()
        thread_logger = logging.getLogger(threadname)
        thread_logger.info('distributed_logger starting')
        while True:
            record = logging_q.get(block=True)
            if record == 'TER':
                thread_logger.info('received {}'.format(record))
                break
            LOGGER.handle(record)

            