# coding: utf-8
import os
import yaml
import socket
import time
import sys
import logging


def load_config(data_path):
    f = open(data_path, 'r')
    conf = yaml.load(f)
    f.close()
    return conf

configure_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../conf/train_config.yaml')
config_ = load_config(configure_path)

