import os

# Main constants
SOURCES_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SOURCES_DIR)

# Config constants
CONFIG_DIR = os.path.join(SOURCES_DIR, 'configs')
CONFIG_STANDARD = os.path.join(CONFIG_DIR, 'standard_config.yml')
CONFIG_VOC = os.path.join(CONFIG_DIR, 'voc_config.yml')

if os.environ.get('COMPUTERNAME', default='MacBook Air') == '19-002464':
    # Constants for AMC PC
    RUNS_DIR = os.path.join('C:', 'Andrei', 'RUNS')
    DATA_DIR = os.path.join('C:', 'Andrei', 'data')
else:
    # Constants for Mac
    RUNS_DIR = os.path.join(ROOT_DIR, 'runs')
    DATA_DIR = os.path.join(ROOT_DIR, 'data')

VOC_ROOT = os.path.join(DATA_DIR, 'VOC')
