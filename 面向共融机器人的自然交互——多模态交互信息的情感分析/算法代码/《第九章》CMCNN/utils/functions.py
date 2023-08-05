import torch
import numpy as np

def dict_to_str(src_dict):
    dst_str = ""
    for key in src_dict.keys():
        if key != 'CM':
            dst_str += " %s: %.4f " %(key, src_dict[key]) 
    return dst_str