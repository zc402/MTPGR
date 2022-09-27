from mtpgr.config.defaults import get_cfg_defaults
from train import Trainer
from mtpgr.utils.log import log

graphs = ['P', 'R', 'PR', 'CP', 'CPR']
partitions = ['SCPS', 'RHPS']
classes = [33, 9]

for g in graphs:
    for p in partitions:
        for c in classes:
            cfg = get_cfg_defaults()
            cfg.MODEL.GRAPH = g
            cfg.MODEL.STRATEGY = p
            cfg.DATASET.NUM_CLASSES = c
            cfg.DATASET.EPOCHS = 1
            log.debug(f'cfg - g:{g}, p:{p}, c:{c}')
            Trainer.from_config(cfg).train()