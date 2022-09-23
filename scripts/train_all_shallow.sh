#!/bin/bash

python train.py -c shallow/stgcn.yaml 
python train.py -c shallow/stgcn_sparse.yaml 
python train.py -c shallow/mtpgr.yaml
python train.py -c shallow/mtpgr_wo_rot.yaml 
python train.py -c shallow/mtpgr_wo_rot_cam.yaml 
python train.py -c shallow/mtpgr_scps.yaml 
python train.py -c shallow/mtpgr_9classes.yaml
