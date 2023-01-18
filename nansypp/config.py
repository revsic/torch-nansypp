class Config:
    """NANSY++ configurations.
    """
    def __init__(self):
        self.sr = 22050

        # unknown all STFT hyperparameters
        self.mel = 80
        self.mel_hop = 256
        self.mel_win = 1024
        self.mel_win_fn = 'hann'
        self.mel_fmin = 0
        self.mel_fmax = 8000

        # unknown
        # , default negative-slope of nn.LeakyReLU
        self.leak = 0.01
        # , default dropout rate of nn.Transformer
        self.dropout = 0.1

        # Wav2Vec2Wrapper
        self.w2v2_name = 'facebook/wav2vec2-large-xlsr-53'
        self.w2v2_lin = 15

        # FrameLevelSynthesizer
        self.frame_kernels = 3
        self.frame_dilations = [1, 3, 9, 27, 1, 3, 9, 27]
        self.frame_blocks = 2

        # LinguisticEncoder
        self.ling_hiddens = 128
        self.ling_preconv = 2
        self.ling_kernels = [3] * 8 + [1] * 2

        # ConstantQTransform
        self.cqt_hop = 256
        self.cqt_fmin = 32.7
        # self.cqt_fmax = 8000
        self.cqt_bins = 191
        self.cqt_bins_per_octave = 24

        # PitchEncoder
        self.pitch_freq = 160
        self.pitch_prekernels = 7
        self.pitch_kernels = 3
        self.pitch_channels = 128
        self.pitch_blocks = 2
        # unknown
        self.pitch_gru = 256
        # unknown
        self.pitch_hiddens = 256
        self.pitch_f0_bins = 64
        self.pitch_start = 50  # hz
        self.pitch_end = 1000

        # Synthesizer
        self.synth_channels = 64
        self.synth_kernels = 3
        self.synth_dilation_rate = 2
        self.synth_layers = 10
        self.synth_cycles = 3

        # TimberEncoder
        self.timb_global = 192
        self.timb_channels = 512
        self.timb_prekernels = 5
        self.timb_scale = 8
        self.timb_kernels = 3
        self.timb_dilations = [2, 3, 4]
        self.timb_bottleneck = 128
        # NANSY++: 3072
        self.timb_hiddens = 1536
        self.timb_latent = 512
        self.timb_timber = 128
        self.timb_tokens = 50
        # unknown
        self.timb_heads = 8
        # unknown
        self.timb_slerp = 0.5
