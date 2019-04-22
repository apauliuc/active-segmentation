import os
import numpy as np
import torch
import SimpleITK as SiTK

from data import MDSDataLoaders
from helpers.config import ConfigClass
from helpers.utils import binarize_nparray
from models import get_model
from helpers.torch_utils import device
from helpers.metrics import SegmentationMetrics
# noinspection PyProtectedMember
from ignite._utils import convert_tensor


def prepare_batch(batch, device_local=None, non_blocking=False):
    """Prepare batch for training: pass to a device with options

    """
    x, y = batch
    return (convert_tensor(x, device=device_local, non_blocking=non_blocking),
            convert_tensor(y, device=device_local, non_blocking=non_blocking))


def main_predict(config: ConfigClass, load_directory=None, name=None, use_best_model=True):
    if name is None:
        name = config.run_name
    config.data.mode = 'predict'

    # Find model file to load from
    files_list = os.listdir(load_directory)
    model_filename = 'final_model_1.pth'

    fname_pattern = 'best_loss_' if use_best_model else 'final_model_'
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
    segment_metrics = SegmentationMetrics(num_classes=loader_wrapper.num_classes,
                                          threshold=config.binarize_threshold)

    # Predict
    for dir_id in loader_wrapper.dir_list:
        data_loader = loader_wrapper.get_predict_loader(dir_id)

        model.eval()
        segmentation = np.zeros(data_loader.dataset.shape)

        idx = 0

        with torch.no_grad():
            for batch in data_loader:
                x, y = prepare_batch(batch, device_local=device, non_blocking=True)
                y_pred = model(x)

                segment_metrics.update((y_pred, y))

                segmentation[idx:idx + config.data.batch_size_val] = torch.sigmoid(y_pred.cpu()).numpy()
                idx += config.data.batch_size_val

        segmentation = segmentation.squeeze(1)
        segmentation = binarize_nparray(segmentation, threshold=config.binarize_threshold)
        segmentation *= 255
        segmentation = segmentation.astype(np.uint8)
        SiTK.WriteImage(SiTK.GetImageFromArray(segmentation),
                        os.path.join(loader_wrapper.predict_path, dir_id, f'{dir_id}_predicted_{name}.mha'))

    print('Predictions done. Segmentation metrics:')

    metrics = segment_metrics.compute()
    print(metrics)
