class Config:
    """NANSY++ configurations.
    """
    def __init__(self):
        self.sr = None

        # unknown all STFT hyperparameters
        self.mel = 80
        self.mel_hop = 256
        self.mel_win = 1024
        self.mel_win_fn = 'hann'
        self.mel_fmin = 0
        self.mel_fmax = 8000

        # unknown
        self.leak = None
        self.dropout = None

        # Wav2Vec2Wrapper
        self.w2v2_name = 'facebook/wav2vec2-large-xlsr-53'
        self.w2v2_lin = 15
        self.w2v2_channels = 1024

        # FrameLevelSynthesizer
        self.frame_kernels = 3
        self.frame_dilations = [1, 3, 9, 27, 1, 3, 9, 27]
        self.frame_blocks = 2

        # LinguisticEncoder
        self.ling_hiddens = 128
        self.ling_preconv = 2
        self.ling_kernels = [3] * 8 + [1] * 2

        # PitchEncoder
        self.pitch_freq = _
        self.pitch_prekernels = 7
        self.pitch_kernels = 3
        self.pitch_channels = 128
        self.pitch_blocks = 2
        # unknown
        self.pitch_gru = None
        # unknown
        self.pitch_hiddens = None
        self.pitch_f0_bins = 64
        self.pitch_start = 50  # hz
        self.pitch_end = 1000

        # Synthesizer
        self.synth_scale = _
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
        self.timb_heads = None
        # unknown
        self.timb_slerp = None
