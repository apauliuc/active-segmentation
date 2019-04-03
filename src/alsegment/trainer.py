
class Trainer(object):
    save_dir: str

    def __init__(self, config):
        self.data_cfg = config['data']
        self.model_cfg = config['model']
        self.train_cfg = config['training']
