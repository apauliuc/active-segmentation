import os
import numpy as np

import torch
# noinspection PyPep8Naming
import SimpleITK as sitk

from definitions import DATA_DIR, DATA_DIR_AT_AMC, RUNS_DIR
from alsegment.helpers.types import device
from alsegment.helpers.paths import get_dataset_path
from alsegment.data.prediction_loader import create_prediction_loader
# noinspection PyProtectedMember
from ignite._utils import convert_tensor

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def prepare_batch(batch, device_local=None, non_blocking=False):
    """Prepare batch for training: pass to a device with options

    """
    return convert_tensor(batch, device=device_local, non_blocking=non_blocking)


def make_predictions(model_path, name=''):
    model = torch.load(model_path)
    model.to(device)

    pred_data_path = get_dataset_path(DATA_DIR_AT_AMC, dataset_type='predict')
    dir_list = [x for x in os.listdir(pred_data_path)
                if os.path.isdir(os.path.join(pred_data_path, x))]

    batch_size = 8

    for dir_id in dir_list:
        scan_dir = os.path.join(pred_data_path, dir_id)
        s_name = os.path.join(dir_id, f'{dir_id}_scan.npy')

        data_loader = create_prediction_loader(pred_data_path, s_name, batch_size=batch_size)

        model.eval()
        segmentation = np.zeros(data_loader.dataset.shape)

        idx = 0

        with torch.no_grad():
            for batch in data_loader:
                x = prepare_batch(batch, device_local=device, non_blocking=True)
                y_pred = model(x)
                y_pred = torch.sigmoid(y_pred.cpu())

                segmentation[idx:idx+batch_size] = y_pred.numpy()
                idx += batch_size

        segmentation = segmentation.squeeze(1)
        segmentation *= 255
        segmentation = segmentation.astype(np.uint8)
        sitk.WriteImage(sitk.GetImageFromArray(segmentation),
                        os.path.join(scan_dir, f'{dir_id}_predicted_{name}.mha'))

    print('Done!')


if __name__ == '__main__':
    load_directory = os.path.join(RUNS_DIR, 'UNet_V2_Base32_BCE_Jacc_04-01_13-26-28')
    file_list = os.listdir(load_directory)

    model_load_file = 'best_model_1.pth'
    for f in file_list:
        if 'best_model_' in f:
            model_load_file = f
    model_load_path = os.path.join(load_directory, model_load_file)

    # cfg_path = os.path.join(run_dir_prev, 'old_1.yml')
    # with open(cfg_path) as f:
    #     config = yaml.load(f)

    make_predictions(model_load_path, 'V2_32_BCE_Jacc')
