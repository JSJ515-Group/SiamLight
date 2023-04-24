import argparse
import os

import torch 
import sys        
sys.path.append(os.path.abspath('.')) # 

import argparse  
import os         

import torch   
import sys     
sys.path.append(os.getcwd())  

from nanotrack.core.config import cfg

from nanotrack.models.model_builder import ModelBuilder

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

from thop import profile 

from thop.utils import clever_format 

def main():

    cfg.merge_from_file('E:/dataset/NanoTrack/models/config/config.yaml')
    
    model = ModelBuilder() 

    x = torch.randn(1, 3, 255, 255)
    zf = torch.randn(1, 3, 127, 127) 

    model.template(zf)  
    
    flop, params = profile(model, inputs=(x,), verbose = False)

    flop, params = clever_format([flop, params], "%.3f")
    
    print('overall flop is ', flop)
    
    print('overall params is ', params)

if __name__ == '__main__': 
    main()
