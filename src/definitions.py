import os

SOURCES_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SOURCES_DIR)
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RUNS_DIR = os.path.join(ROOT_DIR, 'runs')
CONFIG_DIR = os.path.join(SOURCES_DIR, 'configs')

DATA_DIR_AT_AMC = os.path.join('C:', 'Andrei', 'data')
VOC_ROOT = os.path.join(DATA_DIR_AT_AMC, 'VOC')

CONFIG_STANDARD = os.path.join(CONFIG_DIR, 'standard_config.yml')
