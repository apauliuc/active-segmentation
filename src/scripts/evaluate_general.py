import os
import torch

from definitions import DATA_DIR
from data import get_dataloaders
from helpers.config import ConfigClass
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


def evaluate_one_pass(config: ConfigClass, model):
    device = torch.device(f'cuda:{config.gpu_node}' if torch.cuda.is_available() else 'cpu')

    # Create dataloader class
    loader_class = get_dataloaders(config.data)
    segment_metrics = SegmentationMetrics(num_classes=loader_class.num_classes,
                                          threshold=config.binarize_threshold)

    # Evaluate model
    model.eval()

    with torch.no_grad():
        for batch in loader_class.evaluation_loader:
            x, y = batch
            x = x.to(device=device, non_blocking=True)
            y = y.to(device=device, non_blocking=True)

            y_pred = model(x)

            segment_metrics.update((y_pred, y))

    metrics = segment_metrics.compute()
    return metrics


def evaluate_monte_carlo(config: ConfigClass, model):
    device = torch.device(f'cuda:{config.gpu_node}' if torch.cuda.is_available() else 'cpu')
    mc_passes = config.prediction.mc_passes

    # Create dataloader class
    loader_class = get_dataloaders(config.data)
    segment_metrics = SegmentationMetrics(num_classes=loader_class.num_classes,
                                          threshold=config.binarize_threshold)

    # Evaluate model
    model.eval()
    model.apply(apply_dropout)

    with torch.no_grad():
        for batch in loader_class.evaluation_loader:
            x, y = batch
            x = x.to(device=device, non_blocking=True)
            y = y.to(device=device, non_blocking=True)

            summed_y = torch.zeros_like(y)

            for idx in range(mc_passes):
                y_pred = model(x)
                summed_y += torch.sigmoid(y_pred)

            averaged_y = summed_y / mc_passes
            segment_metrics.update((averaged_y, y), process=False)

    metrics = segment_metrics.compute()
    return metrics


def evaluate_ensemble(config: ConfigClass, models_list):
    device = torch.device(f'cuda:{config.gpu_node}' if torch.cuda.is_available() else 'cpu')

    # Create dataloader class
    loader_class = get_dataloaders(config.data)
    segment_metrics = SegmentationMetrics(num_classes=loader_class.num_classes,
                                          threshold=config.binarize_threshold)

    # Evaluate models
    for model in models_list:
        model.eval()

    with torch.no_grad():
        for batch in loader_class.evaluation_loader:
            x, y = batch
            x = x.to(device=device, non_blocking=True)
            y = y.to(device=device, non_blocking=True)

            averaged_y = torch.zeros_like(y)

            for model in models_list:
                y_pred = model(x)
                averaged_y += torch.sigmoid(y_pred)

            averaged_y = averaged_y / len(models_list)
            segment_metrics.update((averaged_y, y), process=False)

    metrics = segment_metrics.compute()
    return metrics


def main_evaluation(config: ConfigClass, load_directory=None, use_best_model=True, verbose=True, save_metrics=True):
    config.data.mode = 'evaluate'
    config.data.path = DATA_DIR

    # Load model
    if config.training.use_ensemble:
        models_list = load_ensemble_models(config, load_directory, use_best_model)

        metrics = evaluate_ensemble(config, models_list)
    else:
        model = load_single_model(config, load_directory, use_best_model)

        if config.prediction.mode == 'single':
            metrics = evaluate_one_pass(config, model)
        else:
            metrics = evaluate_monte_carlo(config, model)
        
    if verbose:
        print(f'Evaluations done. Segmentation metrics: {metrics}')

    if save_metrics:
        with open(os.path.join(load_directory, 'evaluation_results.txt'), 'w+') as f:
            for k, v in metrics.items():
                f.write(f"{k}: {v:.4f}\n")

    return metrics
