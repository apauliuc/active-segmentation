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


def make_predictions(model_path):
    model = torch.load(model_path)
    model.to(device)

    pred_data_path = get_dataset_path(DATA_DIR_AT_AMC, dataset_type='predict')
    scans_list = [x for x in os.listdir(pred_data_path) if 'scan.npy' in x]

    batch_size = 10

    for s_name in scans_list:
        scan_no = s_name.split("_")[0]

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
        sitk.WriteImage(sitk.GetImageFromArray(segmentation),
                        os.path.join(pred_data_path, f'{scan_no}_predicted_comb_losses_255.mha'))

    print('Done!')


if __name__ == '__main__':
    run_dir_prev = os.path.join(RUNS_DIR, 'Combined_losses_03-20_19-43-19')
    # cfg_path = os.path.join(run_dir_prev, 'unet_1.yml')
    model_load_path = os.path.join(run_dir_prev, 'best_model_14.pth')

    # with open(cfg_path) as f:
    #     config = yaml.load(f)

    make_predictions(model_load_path)
