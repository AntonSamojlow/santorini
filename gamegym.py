import logging
import logging.handlers
import multiprocessing
import multiprocessing.queues as mpq
import threading
import os
import datetime
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
            copy_tree(intialmodelpath, self.path.model_folder)
            LOGGER.debug("copied initial model from '{}' to '{}'".format(intialmodelpath, self.path.model_folder))
            with open(self.path.modelinfo_file, 'w+') as f:
                f.write(gymdata.ModelInfo(0).as_json(indent=0))
                LOGGER.debug("created new model info file (iteration count = 0)")
 
    @property
    def path(self):
        return self._path

    def resume(self, runtime_in_sec=None):
        try:
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
            monitoring_q = NamedMultiProcessingQueue("monitoring_q")
            predict_response_qs = { "Selfplayer-{}".format(i) : NamedMultiProcessingQueue("Selfplayer-{} predict_response_q".format(i))
                for i in range(self.config.selfplay.selfplayprocesses)}
            queues = [logging_q, predict_request_q, monitoring_q] + list(predict_response_qs.values())
            
            # threads - configuration and start
            logging_thread = threading.Thread(
                target=self.__class__._centrallogger_worker,
                args=(logging_q,), 
                daemon=True, # note: also execpts 'TER' signal for a regular shutdown
                name="logging")
            inputreader_thread= threading.Thread(
                target=self.__class__._inputreader,
                daemon=True, # note: due to input() blocking, there is no end signaling for this thread
                args=(endevent,),
                name="inputreader")
            monitoring_thread = threading.Thread(
                target=self._monitoring_worker,
                daemon=True, # note: also execpts 'TER' signal for a regular shutdown
                args =(monitoring_q),
                name="monitoring"
            )     
            threads = [logging_thread, inputreader_thread, monitoring_thread]
            
            # processes - configuration
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
                    predict_response_qs)
            selfplayer_procs =[Selfplayer(
                        selfplayername,
                        self.config.selfplay,
                        self.path,
                        monitoring_q,
                        endevent,
                        logging_q,
                        predict_request_q,
                        predict_response_qs[selfplayername],
                        self.graph)          
                for selfplayername in predict_response_qs]
            processes = [trainer_proc, predictor_proc] + selfplayer_procs

            # cleanup of defined ressources
            def cleanup():
                LOGGER.info("endevent was set - starting cleanup of ressources")
                logging_q.put('TER')
                monitoring_q.put('TER')
                
                for p in processes:
                    p.join(10)          
                    if p.is_alive():
                        LOGGER.warning("process {} has not ended - using terminate command".format(p.name))
                        p.terminate()                
                    else:
                        LOGGER.debug("process {} has ended".format(p.name))
                
                for t in threads:
                    if t.isDaemon() and t.is_alive():
                        LOGGER.debug("thread {} is deamonic and alive - skipping join".format(t.name))
                    else:
                        t.join(5)
                        if t.is_alive():
                            LOGGER.warning("thread {} has not ended".format(t.name))
                        else:
                            LOGGER.debug("thread {} has ended".format(t.name))

                for q in queues:
                    if not q.empty():
                        LOGGER.debug("{} items on queue {} - emptying".format(q.qsize(), q.name))
                        while not q.empty():
                            q.get()
                    if isinstance(q, mpq.Queue):
                        q.close()
                        q.join_thread()            
                        LOGGER.debug("queue {} joined".format(q.name))
                LOGGER.info("cleanup finished, process ending")
            
            # ----------------------------------------------------------------------
            # 2. Run - start threads, process and handle until endevent
            # ----------------------------------------------------------------------
            # start threads
            LOGGER.debug("starting threads")
            for t in threads:
                LOGGER.debug("starting {} ...".format(t.name))
                t.start()
                while not t.is_alive():
                    sleep(0.1)
                LOGGER.debug("{} is alive".format(t.name))

            # start processes
            LOGGER.debug("starting processes")
            for p in processes:
                p.start()

            LOGGER.info("idling until endevent is signaled...")
            current_threadname = threading.currentThread().getName()

            t0_livesignal = 0
            t0_start = time()
            while not endevent.is_set():
                if time()-t0_livesignal > self.config.freq_livesignal:
                    t0_livesignal = time()
                    data = (
                        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        __name__,
                        gymdata.MonitoringLabel.LIVESIGNAL,
                        {p.name: p.is_alive() for p in processes})
                    monitoring_q.put(data)
                sleep(1)

                if runtime_in_sec != None:
                    if time()-t0_start > runtime_in_sec:
                        endevent.set()

            # ----------------------------------------------------------------------
            # 3. Endphase - cleaning up ressources
            # ----------------------------------------------------------------------
            
            cleanup()
            return
        except Exception as exc:
            LOGGER.error("exception: {}".format(exc))
            LOGGER.info("signaling endevent and attempting cleanup")            
            try:
                endevent.set()
                cleanup()
            except Exception as finalexc:    
                LOGGER.error("exception: {}".format(finalexc))
                LOGGER.error("terminating without cleanup")
                LOGGER.error("some threads or processes might still be running")
           
    def _monitoring_worker(self, inbox_q: mpq.Queue):
        """Entry for a thread managing data and signals for monitoring"""
        threadname = threading.currentThread().getName()
        logger = LOGGER.getChild(threadname)
        logger.setLevel(logging.DEBUG)
        logger.info(f"{threadname} started")
        while True:
            data = inbox_q.get(block=True)
            if data == 'TER':
                logger.info(f"received {data}")
                break
            else:
                logger.debug(f"received {data}")
                timestamp, sender, label, content = data
                
                if label == gymdata.MonitoringLabel.LIVESIGNAL:
                    filepath = f"{self.path.monitoring_folder}/livesignal.json"
                    with open(filepath, 'a+') as f:
                        f.write(f"{timestamp}|{json.dumps(content)}\n")

                if label == gymdata.MonitoringLabel.SELFPLAYSTATS:
                    filepath = f"{self.path.monitoring_folder}/{sender}_stats.json"
                    with open(filepath, 'a+') as f:
                        f.write(f"{timestamp}|{json.dumps(content)}\n")
    
    @staticmethod
    def _inputreader_worker(endevent : multiprocessing.Event):
        """Entry for a thread waiting for keyboard input."""
        threadname = threading.currentThread().getName()
        logger = LOGGER.getChild(threadname)
        logger.setLevel(logging.DEBUG)
        logger.info('{} started'.format(threadname))
        while True:
            text = input()
            logger.info("registered input '{}'".format(text))
            if text == "exit":
                endevent.set()
                return
           
    @staticmethod
    def _centrallogger_worker(logging_q : mpq.Queue):
        """Entry for a thread handling logging from subprocesses via queues."""
        threadname = threading.currentThread().getName()
        thread_logger = LOGGER.getChild(threadname)
        thread_logger.info('{} started'.format(threadname))
        while True: # only calling process terminates - keeping logging alive 
            record = logging_q.get(block=True)
            if record == 'TER':
                thread_logger.info('received {}'.format(record))
                break
            LOGGER.handle(record)
            qsize = logging_q.qsize()
            if qsize > 10:
                thread_logger.warning(f"logging_q has size {qsize}")
                if logging_q.full():
                    thread_logger.warning(f"logging_q is full")
           

            