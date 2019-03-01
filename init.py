
class InitParser(object):
    def __init__(self):
        # gpu setting
        self.gpu_id = 2
        self.batch_size = 1
        self.train_ratio = 0.8

        # optimizer setting
        self.lr = 1e-4
        self.momentum = 0.9
        self.weight_decay = 1e-4

        # train setting
        self.num_epoch = 50
        self.init_epoch = 1
        self.is_load = False

        # path setting
        self.output_path = "../output/h_Net/"
        self.data_path = "../data/Original"
        self.load_path = "../output/FCN3D/Network_{}.pth.gz".format(self.init_epoch-1)

