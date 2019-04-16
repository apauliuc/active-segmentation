import os
import numpy as np
import torch
import SimpleITK as SiTK

from data import MDSDataLoaders
from helpers.config import ConfigClass
from helpers.utils import binarize_nparray
from models import get_model
from helpers.torch_utils import device
# noinspection PyProtectedMember
from ignite._utils import convert_tensor


def prepare_batch(batch, device_local=None, non_blocking=False):
    """Prepare batch for training: pass to a device with options

    """
    return convert_tensor(batch, device=device_local, non_blocking=non_blocking)


def main_predict(config: ConfigClass, load_directory=None, name=None, use_best_model=True):
    if name is None:
        name = config.run_name
    config.data.mode = 'predict'

    # Find model file to load from
    files_list = os.listdir(load_directory)
    model_filename = 'best_model_1.pth'

    fname_pattern = 'best_model_' if use_best_model else 'final_model_'
    for f in files_list:
        if fname_pattern in f:
            model_filename = f

    model_filepath = os.path.join(load_directory, model_filename)

    # Load model
    model = get_model(config.model)
    model.load_state_dict(torch.load(model_filepath))
    model.to(device)

    # Create dataloader wrapper
    loader_wrapper = MDSDataLoaders(config.data)

    # Predict
    for dir_id in loader_wrapper.dir_list:
        s_name = os.path.join(dir_id, f'{dir_id}_scan.npy')

        data_loader = loader_wrapper.get_predict_loader(s_name)

        model.eval()
        segmentation = np.zeros(data_loader.dataset.shape)

        idx = 0

        with torch.no_grad():
            for batch in data_loader:
                x = prepare_batch(batch, device_local=device, non_blocking=True)
                y_pred = model(x)
                y_pred = torch.sigmoid(y_pred.cpu())

                segmentation[idx:idx + config.data.batch_size_val] = y_pred.numpy()
                idx += config.data.batch_size_val

        segmentation = segmentation.squeeze(1)
        segmentation = binarize_nparray(segmentation, threshold=config.binarize_threshold)
        segmentation *= 255
        segmentation = segmentation.astype(np.uint8)
        SiTK.WriteImage(SiTK.GetImageFromArray(segmentation),
                        os.path.join(loader_wrapper.predict_path, dir_id, f'{dir_id}_predicted_{name}.mha'))

    print('Done!')
