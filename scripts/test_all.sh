#!/bin/bash

python test.py -c stgcn.yaml 
python test.py -c stgcn_sparse.yaml 
python test.py -c mtpgr.yaml
python test.py -c mtpgr_wo_rot.yaml 
python test.py -c mtpgr_wo_rot_cam.yaml 
python test.py -c mtpgr_scps.yaml 
python test.py -c mtpgr_9classes.yaml