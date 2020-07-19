"""
Collectiong of dataclasses and structures used by GameGym and its subprocesses.
"""

import logging
import json
import enum
from dataclasses import dataclass, is_dataclass


class MonitoringLabel(enum.Enum):
    LIVESIGNAL = enum.auto()
    SELFPLAYSTATS = enum.auto()


class JSONSerializableDataClass():
    """Custom JSON serialization support for (nested) dataclasses"""
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
        def default(self, obj):  # pylint: disable=E0202
            if is_dataclass(obj):
                content = {obj.__class__.__name__: True}
                content.update(obj.__dict__)
                return content
            return json.JSONEncoder.default(self, obj)


@dataclass
class LoggingConfig(JSONSerializableDataClass):
    """Configuration for the logging setup of a GameGym-subprocess"""
    sizelimit_MB: float = 10
    backupCount: int = 9
    loglevel: str = "DEBUG"
    tf_loglevel: str = 'ERROR'

    def getRotatingFileHandler(self, logpath: str):
        rfh = logging.handlers.RotatingFileHandler(
            logpath,
            maxBytes=self.sizelimit_MB * 1000000,
            backupCount=self.backupCount)
        rfh.setLevel(self.loglevel)
        rfh.setFormatter(
            logging.Formatter(
                '%(asctime)s [%(name)s] %(levelname)s: %(message)s'))
        return rfh


@dataclass
class PredictConfig(JSONSerializableDataClass):
    """Configuration for the Predictor subprocess"""
    batchsize: int
    trygetbatchsize_timeout = 0.1
    logging: LoggingConfig = LoggingConfig()
    use_gpu: bool = False
    gpu_memorylimit: int = None


@dataclass
class SelfPlayConfig(JSONSerializableDataClass):
    """Configuration for the Selfplayer subprocess"""
    selfplayprocesses: int = 1
    searchthreadcount: int = 10
    searchcount: int = 100
    exploration_const: float = 2.0
    virtualloss: float = 0.1
    sleeptime_blocked_select: float = 0.1
    temperature: float = 1.0
    gamelog_dump_threshold: int = 10
    freq_statssignal: int = 60
    logging: LoggingConfig = LoggingConfig()


@dataclass
class EvaluateConfig(JSONSerializableDataClass):
    """Configuration for the Predictor subprocess"""
    model1_savefolderpath: str
    model1_weightsfolderpath: str
    model2_savefolderpath: str
    model2_weightsfolderpath: str
    playconfig: SelfPlayConfig = SelfPlayConfig()
    logging: LoggingConfig = LoggingConfig()


@dataclass
class TrainConfig(JSONSerializableDataClass):
    """Configuration for the Trainer subprocess"""

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
    """Configuration for the GameGym mainprocess"""
    predict: PredictConfig
    selfplay: SelfPlayConfig
    train: TrainConfig
    freq_livesignal: int = 60
    logging: LoggingConfig = LoggingConfig()


@dataclass
class GymPath(JSONSerializableDataClass):
    """Represent the fixed subfolder and file structure of a GamGym session, based on the sessions basefolder"""
    basefolder: str

    @property
    def config_file(self) -> str:
        return f"{self.basefolder}/config.json"

    @property
    def modelinfo_file(self) -> str:
        return f"{self.basefolder}/model/info.json"

    @property
    def model_folder(self) -> str:
        return f"{self.basefolder}/model/"

    @property
    def weights_folder(self) -> str:
        return f"{self.basefolder}/weights/"

    @property
    def trainhistories_folder(self) -> str:
        return f"{self.basefolder}/trainhistories/"

    @property
    def gamerecordpool_folder(self) -> str:
        return f"{self.basefolder}/gamerecordpool/"

    @property
    def log_folder(self) -> str:
        return f"{self.basefolder}/logs/"

    @property
    def monitoring_folder(self) -> str:
        return f"{self.basefolder}/monitoring/"

    @property
    def subfolders(self) -> list:
        return [
            self.model_folder, self.gamerecordpool_folder, self.log_folder,
            self.weights_folder, self.trainhistories_folder,
            self.monitoring_folder
        ]


@dataclass
class ModelInfo(JSONSerializableDataClass):
    """Information about the currently active model of the GameGym session"""
    iterationNr: int
