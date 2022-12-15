class Config:
    """Discriminator configurations.
    """
    def __init__(self):
        self.periods = [2, 3, 5, 7, 11]
        self.channels = [32, 128, 512, 1024]
        self.kernels = 5
        self.strides = 3
        self.postkernels = 3
        self.leak = 0.1
