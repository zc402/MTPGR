from mtpgr.config.defaults import get_cfg_defaults
from test import Tester
from train import Trainer
from mtpgr.utils.log import log

def debug():
    cfg = get_cfg_defaults()
    cfg.MODEL.GRAPH = 'CPR'
    cfg.MODEL.STRATEGY = 'RHPS'
    cfg.DATASET.NUM_CLASSES = 33
    cfg.DATASET.EPOCHS = 50
    # Trainer.from_config(cfg).train()
    Tester.from_config(cfg).test()

debug()