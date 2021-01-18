# Options
class SimpleOpt:
    def __init__(self):
        self.input_size = 128
        self.emb_size = 128
        self.sample_size = 1024
        self.max_epochs = 2000
        self.lr = 0.01
        self.gpu = '0'
        self.DATA = 'google'
        self.output_name = 'fastgae-no_pn-relu'


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
