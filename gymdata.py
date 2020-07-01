import logging
import json
import enum
from dataclasses import dataclass, is_dataclass

class MonitoringLabel(enum.Enum):
    LIVESIGNAL = enum.auto()
    SELFPLAYSTATS = enum.auto()
    

class JSONSerializableDataClass():
    """Provides JSON serialization support for (nested) dataclasses"""

    def as_json(self, indent=2):           
        return json.dumps(self, cls=self.DataClassEncoder, indent=indent)

    @classmethod
    def from_json(cls, string):
        def decode(dct):
            for subclass in JSONSerializableDataClass.__subclasses__():
                if subclass.__name__ in dct:   
                    content = dict(dct)
                    content.pop(subclass.__name__)   
                    return subclass(*content.values())
            return dct 
        return json.loads(string, object_hook=decode)

    class DataClassEncoder(json.JSONEncoder):
        """Custom Encoder"""
        def default(self, obj):  # pylint: disable=E0202
            if is_dataclass(obj):
                content = {obj.__class__.__name__ : True}
                content.update(obj.__dict__)
                return content
            return json.JSONEncoder.default(self, obj)


@dataclass
class LoggingConfig(JSONSerializableDataClass):  
    sizelimit_MB: float = 10
    backupCount: int = 9
    loglevel: str = "DEBUG"
    tf_loglevel: str = 'ERROR'

    def getRotatingFileHandler(self, logpath: str):
        rfh = logging.handlers.RotatingFileHandler(logpath, 
            maxBytes=self.sizelimit_MB*1000000, backupCount=self.backupCount)
        rfh.setLevel(self.loglevel)     
        rfh.setFormatter(logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s'))
        return rfh

@dataclass
class PredictConfig(JSONSerializableDataClass):
    batchsize: int
    trygetbatchsize_timeout = 0.1   
    logging: LoggingConfig = LoggingConfig()
    use_gpu: bool = True
    gpu_memorylimit: int = None

@dataclass
class SelfPlayConfig(JSONSerializableDataClass):
    selfplayprocesses :  int
    searchthreadcount : int
    searchcount : int
    exploration_const: float = 2.0 
    virtualloss: float = 0.1
    sleeptime_blocked_select: float = 0.1
    temperature: float = 1.0
    gamelog_dump_threshold: int = 10
    freq_statssignal: int = 60
    logging: LoggingConfig = LoggingConfig()
    
@dataclass
class TrainConfig(JSONSerializableDataClass):
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
class GymConfig(JSONSerializableDataClass):
    predict: PredictConfig
    selfplay: SelfPlayConfig 
    train: TrainConfig
    freq_livesignal: int = 60
    logging: LoggingConfig = LoggingConfig()

@dataclass
class GymPath(JSONSerializableDataClass):
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
class ModelInfo(JSONSerializableDataClass):
    iterationNr: int


