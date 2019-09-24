import os
import numpy as np
import torch
import SimpleITK as SiTK

from definitions import DATA_DIR
from data import MDSDataLoaders
from helpers.config import ConfigClass
from helpers.utils import binarize_nparray
from helpers.metrics import SegmentationMetrics
from helpers.torch_utils import apply_dropout
from scripts.evaluate_general import load_single_model, load_ensemble_models


def mds_evaluate_one_pass(config: ConfigClass, model, name):
    device = torch.device(f'cuda:{config.gpu_node}' if torch.cuda.is_available() else 'cpu')

    # Create dataloader wrapper
    loader_wrapper = MDSDataLoaders(config.data)
    segment_metrics = SegmentationMetrics(num_classes=loader_wrapper.num_classes,
                                          threshold=config.binarize_threshold)

    # Evaluate
    for dir_id in loader_wrapper.dir_list:
        data_loader = loader_wrapper.get_evaluation_loader(dir_id)

        model.eval()
        segmentation = np.zeros(data_loader.dataset.shape)

        idx = 0

        with torch.no_grad():
            for batch in data_loader:
                x, y = batch
                x = x.to(device=device, non_blocking=True)
                y = y.to(device=device, non_blocking=True)

                y_pred = model(x)

                segment_metrics.update((y_pred, y))

                segmentation[idx:idx + config.data.batch_size_val] = torch.sigmoid(y_pred.cpu()).numpy()
                idx += config.data.batch_size_val

        save_segmentation_to_file(segmentation, config.binarize_threshold, loader_wrapper.predict_path,
                                  dir_id, name)

    metrics = segment_metrics.compute()
    return metrics


def mds_evaluate_monte_carlo(config: ConfigClass, model, name):
    device = torch.device(f'cuda:{config.gpu_node}' if torch.cuda.is_available() else 'cpu')

    # Create dataloader wrapper
    loader_wrapper = MDSDataLoaders(config.data)
    segment_metrics = SegmentationMetrics(num_classes=loader_wrapper.num_classes,
                                          threshold=config.binarize_threshold)

    # Evaluate
    for dir_id in loader_wrapper.dir_list:
        data_loader = loader_wrapper.get_evaluation_loader(dir_id)

        model.eval()
        model.apply(apply_dropout)

        segmentation = torch.zeros(data_loader.dataset.shape)
        ground_truth = torch.zeros_like(segmentation)

        idx = 0

        with torch.no_grad():
            for batch in data_loader:
                x, y = batch
                x = x.to(device=device, non_blocking=True)

                for _ in range(config.prediction.mc_passes):
                    y_pred = model(x)
                    segmentation[idx:idx + config.data.batch_size_val] += torch.sigmoid(y_pred).cpu()

                ground_truth[idx:idx + config.data.batch_size_val] = y
                idx += config.data.batch_size_val

        segmentation = segmentation / config.prediction.mc_passes
        segment_metrics.update((segmentation, ground_truth), process=False)

        segmentation = segmentation.numpy()

        save_segmentation_to_file(segmentation, config.binarize_threshold, loader_wrapper.predict_path,
                                  dir_id, name)

    metrics = segment_metrics.compute()
    return metrics


def mds_evaluate_ensemble(config: ConfigClass, models_list, name):
    device = torch.device(f'cuda:{config.gpu_node}' if torch.cuda.is_available() else 'cpu')

    # Create dataloader wrapper
    loader_wrapper = MDSDataLoaders(config.data)
    segment_metrics = SegmentationMetrics(num_classes=loader_wrapper.num_classes,
                                          threshold=config.binarize_threshold)

    # Evaluate
    for dir_id in loader_wrapper.dir_list:
        data_loader = loader_wrapper.get_evaluation_loader(dir_id)

        for m in models_list:
            m.eval()

        segmentation = torch.zeros(data_loader.dataset.shape)
        ground_truth = torch.zeros_like(segmentation)

        idx = 0

        with torch.no_grad():
            for batch in data_loader:
                x, y = batch
                x = x.to(device=device, non_blocking=True)

                for model in models_list:
                    y_pred = model(x)
                    segmentation[idx:idx + config.data.batch_size_val] += torch.sigmoid(y_pred).cpu()

                ground_truth[idx:idx + config.data.batch_size_val] = y
                idx += config.data.batch_size_val

        segmentation = segmentation / len(models_list)
        segment_metrics.update((segmentation, ground_truth), process=False)

        segmentation = segmentation.numpy()

        save_segmentation_to_file(segmentation, config.binarize_threshold, loader_wrapper.predict_path,
                                  dir_id, name)

    metrics = segment_metrics.compute()
    return metrics


def save_segmentation_to_file(segmentation, threshold, path, dir_id, name):
    segmentation = segmentation.squeeze(1)
    segmentation = binarize_nparray(segmentation, threshold=threshold)
    segmentation *= 255
    segmentation = segmentation.astype(np.uint8)
    SiTK.WriteImage(SiTK.GetImageFromArray(segmentation),
                    os.path.join(path, dir_id, f'{dir_id}_predicted_{name}.mha'))


def main_evaluation_mds(config: ConfigClass, load_directory=None, name=None, use_best_model=True):
    if name is None:
        name = config.run_name
    config.data.mode = 'evaluate'
    config.data.path = DATA_DIR
    config.data.batch_size_val = 12

    # Load model
    if config.training.use_ensemble:
        models_list = load_ensemble_models(config, load_directory, use_best_model)

        metrics = mds_evaluate_ensemble(config, models_list, name)
    else:
        model = load_single_model(config, load_directory, use_best_model)

        if config.prediction.mode == 'single':
            metrics = mds_evaluate_one_pass(config, model, name)
        else:
            metrics = mds_evaluate_monte_carlo(config, model, name)

    print('Evaluations done. Segmentation metrics:')
    print(metrics)

    with open(os.path.join(load_directory, 'evaluation_results.txt'), 'w+') as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
