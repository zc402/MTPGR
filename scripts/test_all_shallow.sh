#!/bin/bash

python test.py -c shallow/stgcn.yaml 
python test.py -c shallow/stgcn_sparse.yaml 
python test.py -c shallow/mtpgr.yaml
python test.py -c shallow/mtpgr_wo_rot.yaml 
python test.py -c shallow/mtpgr_wo_rot_cam.yaml 
python test.py -c shallow/mtpgr_scps.yaml 
python test.py -c shallow/mtpgr_9classes.yaml