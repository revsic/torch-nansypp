from speechset.config import Config as DataConfig
from disc.config import Config as DiscConfig
from nansypp.config import Config as ModelConfig


class TrainConfig:
    """Configuration for training loop.
    """
    def __init__(self, sr: int, hop: int):
        """Initializer.
        Args:
            sr: sample rate.
            hop: stft hop length.
        """
        # optimizer
        self.learning_rate_g = 1e-4
        self.learning_rate_d = 2e-4
        # default beta values
        self.beta1 = 0.9
        self.beta2 = 0.999
        # huber norm
        # unknown
        self.delta = 1.0
        # content loss warmup
        self.content_start = 1e-5
        self.content_end = 10
        self.content_adj = 10
        # unknown
        self.candidates = 15

        # augment
        self.num_code = 32
        self.formant_shift = 1.4
        self.pitch_shift = 2.
        self.pitch_range = 1.5
        self.cutoff_lowpass = 60
        self.cutoff_highpass = 10000
        self.q_min = 2
        self.q_max = 5
        self.num_peak = 8
        self.g_min = -12
        self.g_max = 12
        # pitch consistency
        self.cqt_shift_min = -12
        self.cqt_shift_max = 12
        # linguistic informations
        self.kappa = 0.1

        # objective
        # 16khz sr, default win=[1920, 320, 80], hop=[640, 80, 40] in NSF
        self.wins = [2048, 512, 128]
        self.hops = [512, 128, 32]

        # loader settings
        self.batch = 16
        self.shuffle = True
        self.num_workers = 4
        self.pin_memory = True

        # train iters
        self.epoch = 1000

        # segment length
        sec = 1
        self.seglen = int(sr * sec) // hop * hop

        # path config
        self.log = './log'
        self.ckpt = './ckpt'

        # model name
        self.name = 't1'

        # commit hash
        self.hash = 'unknown'


class Config:
    """Integrated configuration.
    """
    def __init__(self):
        self.data = DataConfig(batch=None)
        self.train = TrainConfig(self.data.sr, self.data.hop)
        self.model = ModelConfig()
        self.disc = DiscConfig()

    def validate(self):
        assert self.data.sr == self.model.sr, \
            'inconsistent data and model settings'

    def dump(self):
        """Dump configurations into serializable dictionary.
        """
        return {k: vars(v) for k, v in vars(self).items()}

    @staticmethod
    def load(dump_):
        """Load dumped configurations into new configuration.
        """
        conf = Config()
        for k, v in dump_.items():
            if hasattr(conf, k):
                obj = getattr(conf, k)
                load_state(obj, v)
        return conf


def load_state(obj, dump_):
    """Load dictionary items to attributes.
    """
    for k, v in dump_.items():
        if hasattr(obj, k):
            setattr(obj, k, v)
    return obj
