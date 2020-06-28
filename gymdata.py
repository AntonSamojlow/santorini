import logging
import json
import enum
from dataclasses import dataclass

class MonitoringLabel(enum.Enum):
    LIVESIGNAL = enum.auto()
    SELFPLAYSTATS = enum.auto()
    

@dataclass
class LoggingConfig():  
    sizelimit_MB: float = 1
    backupCount: int = 9
    loglevel: str = "DEBUG"

    def getRotatingFileHandler(self, logpath: str):
        rfh = logging.handlers.RotatingFileHandler(logpath, 
            maxBytes=self.sizelimit_MB*1000000, backupCount=self.backupCount)
        rfh.setLevel(self.loglevel)     
        rfh.setFormatter(logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s'))
        return rfh

@dataclass
class PredictConfig():
    batchsize: int
    trygetbatchsize_timeout = 0.1
    logging: LoggingConfig = LoggingConfig()
    use_gpu: bool = True
    gpu_memorylimit: int = None

@dataclass
class SelfPlayConfig():
    selfplayprocesses :  int
    searchthreadcount : int
    searchcount : int
    exploration_const: float = 2.0 
    virtualloss: float = 0.1
    sleeptime_blocked_select: float = 0.1
    temperature: float = 1.0
    record_dumpbatchsize: int = 100
    freq_statssignal: int = 60
    logging: LoggingConfig = LoggingConfig()
    
@dataclass
class TrainConfig():
    epochs: int    
    batchsize: int
    min_samplecount: int
    max_samplecount: int
    max_sampleage: int = 3
    validation_split: float = 0.2
    logging: LoggingConfig = LoggingConfig()
    use_gpu: bool = False
    gpu_memorylimit: int = None

@dataclass
class GymConfig():
    predict: PredictConfig
    selfplay: SelfPlayConfig 
    train: TrainConfig
    freq_livesignal: int = 60
    logging: LoggingConfig = LoggingConfig()

@dataclass
class GymPath():
    basefolder : str

    @property
    def config_file(self) ->str:
        return "{}/config.json".format(self.basefolder)  
    @property
    def modelinfo_file(self) ->str:
        return "{}/model/info.json".format(self.basefolder)
    @property
    def model_folder(self) ->str:
        return "{}/model/".format(self.basefolder)
    @property
    def weights_folder(self) ->str:
        return "{}/weights/".format(self.basefolder)
    @property
    def trainhistories_folder(self) ->str:
        return "{}/trainhistories/".format(self.basefolder)
    @property
    def gamerecordpool_folder(self) ->str:
        return "{}/gamerecordpool/".format(self.basefolder)
    @property
    def log_folder(self) ->str:
        return "{}/logs/".format(self.basefolder)
    @property
    def monitoring_folder(self) ->str:
        return "{}/monitoring/".format(self.basefolder)

    @property
    def subfolders(self) -> list:
        return [
            self.model_folder, 
            self.gamerecordpool_folder, 
            self.log_folder, 
            self.weights_folder, 
            self.trainhistories_folder,
            self.monitoring_folder]

@dataclass
class ModelInfo():
    iterationNr: int

    def as_json(self, indent=2):
        class GameGraphEncoder(json.JSONEncoder):
            """Custom Encoder"""
            def default(self, obj):  # pylint: disable=E0202
                if isinstance(obj, ModelInfo):        
                    return {
                        obj.__class__.__name__: True,                        
                        'iterationNr' : obj.iterationNr}
                return json.JSONEncoder.default(self, obj)
        return json.dumps(self, cls=GameGraphEncoder, indent=indent)

    @classmethod
    def from_json(cls, string) -> 'ModelInfo':
        def decode(dct):
            if cls.__name__ in dct:      
                return cls(
                    iterationNr= int(dct['iterationNr']))                   
            return dct
        return json.loads(string, object_hook=decode)    
