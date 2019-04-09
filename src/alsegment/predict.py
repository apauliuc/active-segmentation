import os
import yaml
import numpy as np
import torch
import SimpleITK as SiTK

from alsegment.data.prediction_loader import create_prediction_loader
from alsegment.helpers.paths import get_dataset_path
from alsegment.models import get_model
from definitions import RUNS_DIR
from alsegment.helpers.types import device
# noinspection PyProtectedMember
from ignite._utils import convert_tensor


def prepare_batch(batch, device_local=None, non_blocking=False):
    """Prepare batch for training: pass to a device with options

    """
    return convert_tensor(batch, device=device_local, non_blocking=non_blocking)


def prediction_main(run_dir_name, config, name='', use_best_model=True):
    # Find model file to load from
    load_directory = os.path.join(RUNS_DIR, run_dir_name)
    files_list = os.listdir(load_directory)
    model_filename = 'best_model_1.pth'

    fname_pattern = 'best_model_' if use_best_model else 'final_model_'
    for f in files_list:
        if fname_pattern in f:
            model_filename = f

    model_filepath = os.path.join(load_directory, model_filename)

    # Load model
    model = get_model(config['model'])
    model.load_state_dict(torch.load(model_filepath))
    model.to(device)

    # Create dataloader
    data_cfg = config['data']
    data_path = get_dataset_path(data_cfg['path'], data_cfg['dataset'], 'predict')
    dir_list = [x for x in os.listdir(data_path)
                if os.path.isdir(os.path.join(data_path, x))]

    batch_size = 8

    # Predict
    for dir_id in dir_list:
        scan_dir = os.path.join(data_path, dir_id)
        s_name = os.path.join(dir_id, f'{dir_id}_scan.npy')

        data_loader = create_prediction_loader(data_path, s_name, batch_size=batch_size)

        model.eval()
        segmentation = np.zeros(data_loader.dataset.shape)

        idx = 0

        with torch.no_grad():
            for batch in data_loader:
                x = prepare_batch(batch, device_local=device, non_blocking=True)
                y_pred = model(x)
                y_pred = torch.sigmoid(y_pred.cpu())

                segmentation[idx:idx + batch_size] = y_pred.numpy()
                idx += batch_size

        segmentation = segmentation.squeeze(1)
        segmentation *= 255
        segmentation = segmentation.astype(np.uint8)
        SiTK.WriteImage(SiTK.GetImageFromArray(segmentation),
                        os.path.join(scan_dir, f'{dir_id}_predicted_{name}.mha'))

    print('Done!')
