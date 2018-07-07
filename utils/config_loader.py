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

config_ = load_config('./conf/train_config.yaml')