import logging

import torch

from countFLOPS import count_model_flops, _calc_width

from config.config_FP32 import config as cfg

if __name__ == "__main__":

    backbone = ... # load_your_pytorch_model

    flops = count_model_flops(backbone)
    width = _calc_width(backbone)

