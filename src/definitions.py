import os

# Main constants
SOURCES_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SOURCES_DIR)

# Config constants
CONFIG_DIR = os.path.join(SOURCES_DIR, 'configs')
CONFIG_DEFAULT = os.path.join(CONFIG_DIR, 'default_config.yml')
CONFIG_AL = os.path.join(CONFIG_DIR, 'al_config.yml')
CONFIG_VOC = os.path.join(CONFIG_DIR, 'voc_config.yml')

if os.environ.get('USER') == 'pauliuca' or os.environ.get('USER') == 'andrei':
    # SURFSARA OR MAC
    RUNS_DIR = os.path.join(ROOT_DIR, 'runs')
    DATA_DIR = os.path.join(ROOT_DIR, 'data')
elif os.environ.get('USERNAME') == 'aspauliuc':
    RUNS_DIR = os.path.join('C:', 'Andrei', 'RUNS')
    DATA_DIR = os.path.join('C:', 'Andrei', 'data')

VOC_ROOT = ''
