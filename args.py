# Options
class SimpleOpt:
    def __init__(self):
        self.av_size = 128
        self.emb_size = 64
        self.hidden_size = 128
        self.pool_size = 128
        self.output_size = 64
        self.max_epochs = 20000
        self.lr = 0.0001
        self.gpu = '2'
        self.DATA = 'guarantee_loan'
        self.output_name = 'simplehie'


# using command-line or type-here
class Options():
    def __init__(self):
        self.opt_type = 'simple'
        # self.opt_type = 'command'

    def initialize(self, epoch_num=1):
        opt = SimpleOpt()
        opt.max_epochs = epoch_num
        return opt


def get_options():
    opt = Options()
    opt = opt.initialize()
    return opt
