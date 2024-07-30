#!/usr/bin/env python

# Run the training of the deepsets network given a configuration file.
import argparse
import os
import yaml
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



from fast_deepsets.deepsets.train import main


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--config_file", type=str)
parser.add_argument("--gpu", type=str, default="0")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

with open(args.config_file, "r") as stream:
    config = yaml.load(stream, Loader=yaml.Loader)

main(config)
