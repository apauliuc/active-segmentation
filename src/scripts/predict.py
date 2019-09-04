import os
import numpy as np
import torch
import SimpleITK as SiTK

from definitions import DATA_DIR
from data import MDSDataLoaders
from helpers.config import ConfigClass
from helpers.utils import binarize_nparray
from models import get_model
from helpers.metrics import SegmentationMetrics
from helpers.torch_utils import apply_dropout


def get_files_list(config: ConfigClass, load_directory: str):
    # Find model file to load from
    if config.al_mode:
        al_directory = sorted([x for x in os.listdir(load_directory) if 'Step' in x])[-1]
        load_directory = os.path.join(load_directory, al_directory)
        files_list = os.listdir(load_directory)
    else:
        files_list = os.listdir(load_directory)

    return files_list, load_directory


def load_ensemble_models(config: ConfigClass, load_directory=None, use_best_model=True):
    device = torch.device(f'cuda:{config.gpu_node}' if torch.cuda.is_available() else 'cpu')

    files_list, load_directory = get_files_list(config, load_directory)

    model_filenames = list()
    fname_pattern = 'best_model_' if use_best_model else 'final_model_'
    for f in files_list:
        if fname_pattern in f:
            model_filenames.append(f)

    # Load models
    models = list()
    for f in model_filenames:
        model = get_model(config.model)
        model.load_state_dict(torch.load(os.path.join(load_directory, f)))
        model.to(device)
        models.append(model)

    return models


def load_single_model(config: ConfigClass, load_directory=None, use_best_model=True):
    device = torch.device(f'cuda:{config.gpu_node}' if torch.cuda.is_available() else 'cpu')

    files_list, load_directory = get_files_list(config, load_directory)

    model_filename = 'final_model_1.pth'

    fname_pattern = 'best_model_' if use_best_model else 'final_model_'
    for f in files_list:
        if fname_pattern in f:
            model_filename = f

    model_filepath = os.path.join(load_directory, model_filename)

    # Load model
    model = get_model(config.model)
    model.load_state_dict(torch.load(model_filepath))
    model.to(device)

    return model


def predict_one_pass(config: ConfigClass, model, name):
    device = torch.device(f'cuda:{config.gpu_node}' if torch.cuda.is_available() else 'cpu')

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
                x, y = batch
                x = x.to(device=device, non_blocking=True)
                y = y.to(device=device, non_blocking=True)

                y_pred, *_ = model(x)

                segment_metrics.update((y_pred, y))

                segmentation[idx:idx + config.data.batch_size_val] = torch.sigmoid(y_pred.cpu()).numpy()
                idx += config.data.batch_size_val

        save_segmentation_to_file(segmentation, config.binarize_threshold, loader_wrapper.predict_path,
                                  dir_id, name)

    metrics = segment_metrics.compute()
    return metrics


def predict_monte_carlo(config: ConfigClass, model, name):
    device = torch.device(f'cuda:{config.gpu_node}' if torch.cuda.is_available() else 'cpu')

    # Create dataloader wrapper
    loader_wrapper = MDSDataLoaders(config.data)
    segment_metrics = SegmentationMetrics(num_classes=loader_wrapper.num_classes,
                                          threshold=config.binarize_threshold)

    # Predict
    for dir_id in loader_wrapper.dir_list:
        data_loader = loader_wrapper.get_predict_loader(dir_id)

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


def predict_ensemble(config: ConfigClass, models_list, name):
    device = torch.device(f'cuda:{config.gpu_node}' if torch.cuda.is_available() else 'cpu')

    # Create dataloader wrapper
    loader_wrapper = MDSDataLoaders(config.data)
    segment_metrics = SegmentationMetrics(num_classes=loader_wrapper.num_classes,
                                          threshold=config.binarize_threshold)

    # Predict
    for dir_id in loader_wrapper.dir_list:
        data_loader = loader_wrapper.get_predict_loader(dir_id)

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


def main_predict(config: ConfigClass, load_directory=None, name=None, use_best_model=True):
    if name is None:
        name = config.run_name
    config.data.mode = 'predict'
    config.data.path = DATA_DIR

    # Load model
    if config.training.use_ensemble:
        models_list = load_ensemble_models(config, load_directory, use_best_model)

        metrics = predict_ensemble(config, models_list, name)
    else:
        model = load_single_model(config, load_directory, use_best_model)

        if config.prediction.mode == 'single':
            metrics = predict_one_pass(config, model, name)
        else:
            metrics = predict_monte_carlo(config, model, name)

    print('Predictions done. Segmentation metrics:')
    print(metrics)

    with open(os.path.join(load_directory, 'prediction_results.txt'), 'w+') as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
