
class Config(object):
    """ Wrapper class for various (hyper)parameters. """
    def __init__(self):
        # about the model architecture
        self.image_size = 64
        self.label_length = 6
        self.num_classes = 11
        # about the optimization
        self.batch_size = 32
        self.num_step = 300000
        # about the saver
        self.save_period = 2000
        self.save_dir = './models/'
        self.summary_dir = './logs/'


