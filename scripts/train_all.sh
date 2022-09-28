#!/bin/bash

python train.py -c stgcn.yaml 
# python train.py -c stgcn_sparse.yaml 
python train.py -c stgcn_cam.yaml
python train.py -c mtpgr.yaml
python train.py -c mtpgr_wo_rot.yaml 
python train.py -c mtpgr_wo_rot_cam.yaml 
python train.py -c mtpgr_scps.yaml 
python train.py -c mtpgr_9classes.yaml
