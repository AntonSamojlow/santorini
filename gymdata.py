from dataclasses import dataclass
import logging

@dataclass
class LoggingConfig():  
    sizelimit_kB: float = 100
    backupCount: int = 9
    loglevel: str = "DEBUG"

    def addRotatingFileHandler(self, logger: logging.Logger, logpath: str):
        rfh = logging.handlers.RotatingFileHandler(logpath, 
            maxBytes=self.sizelimit_kB*1000, backupCount=self.backupCount)
        rfh.setLevel(self.loglevel)     
        rfh.setFormatter(logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s'))
        logger.addHandler(rfh)

@dataclass
class PredictConfig():
    batchsize: int
    trygetbatchsize_timeout = 0.1
    logging: LoggingConfig = LoggingConfig()

@dataclass
class SelfPlayConfig():
    searchthreadcount : int
    searchcount : int
    exploration_const: float = 2.0 
    virtualloss: float = 0.1
    sleeptime_blocked_select: float = 0.1
    temperature: float = 1.0
    logging: LoggingConfig = LoggingConfig()
    
@dataclass
class TrainConfig():    
    batchsize: int
    epochs: int
    logging: LoggingConfig = LoggingConfig()

@dataclass
class GymConfig():
    predict: PredictConfig
    selfplay: SelfPlayConfig 
    train: TrainConfig
    logging: LoggingConfig = LoggingConfig()

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

