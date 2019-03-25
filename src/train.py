import os
import yaml
import random
import shutil
import argparse

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter

from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import ModelCheckpoint, Timer, TerminateOnNan
from ignite.metrics import Loss, RunningAverage

from alsegment.models import get_model
from alsegment.losses import get_loss_fn
from definitions import DATA_DIR, DATA_DIR_AT_AMC, CONFIG_STANDARD
from alsegment.helpers.utils import setup_logger, timer_to_str
from alsegment.helpers.types import device
from alsegment.data.dataloader import create_data_loader
from alsegment.helpers.paths import get_dataset_path, get_new_run_path, get_model_optimizer_path

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def train(cfg, save_dir):
    # Initialise writer, logger and configs
    writer = SummaryWriter(log_dir=save_dir)
    logger = setup_logger(save_dir)
    logger.info(f'Saving to folder {save_dir}')

    data_cfg = cfg['data']
    model_cfg = cfg['model']
    train_cfg = cfg['training']

    # Set seed
    if train_cfg['seed'] is not None:
        torch.manual_seed(train_cfg['seed'])
        random.seed(train_cfg['seed'])
        logger.info(f'Seed set on {train_cfg["seed"]}')

    # Create train dataloader
    train_path = get_dataset_path(data_cfg['path'], data_cfg['dataset'], data_cfg['train_split'])
    train_loader = create_data_loader(data_cfg, train_path, batch_size=data_cfg['batch_size'])
    logger.info(f'Train data loader created from {train_path}')

    # Create validation dataloader
    val_path = get_dataset_path(data_cfg['path'], data_cfg['dataset'], data_cfg['val_split'])
    val_loader = create_data_loader(data_cfg, val_path, batch_size=data_cfg['batch_size_val'])
    val_train_loader = create_data_loader(data_cfg, train_path, batch_size=data_cfg['batch_size_val'])
    logger.info(f'Validation data loader created from {val_path}')

    # Create model, loss function and optimizer
    model = get_model(model_cfg).to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=train_cfg['lr'],
                           weight_decay=train_cfg['weight_decay'], amsgrad=train_cfg['amsgrad'])
    criterion = get_loss_fn(train_cfg['loss_fn']).to(device=device)

    logger.info(f'Using model {model} and Loss function: {criterion}')

    # Load model and optimizer if set in config file
    if model_cfg['resume_from'] is not None:
        model_path, optimizer_path = get_model_optimizer_path(model_cfg['resume_from'],
                                                              model_cfg['saved_model'],
                                                              model_cfg['saved_optimizer'])
        model = torch.load(model_path)
        optimizer = torch.load(optimizer_path)
        logger.info(f'Model loaded from {model_path}')
        logger.info(f'Optimizer loaded from {optimizer_path}')

    # Setup learning rate scheduler (to add more options)
    lr_scheduler = None
    if train_cfg['scheduler'] == 'step':
        lr_scheduler = StepLR(optimizer, step_size=train_cfg['lr_cycle'], gamma=0.1)
        logger.info(f'LR scheduler set to type {train_cfg["scheduler"]}')

    # Create engines
    trainer = create_supervised_trainer(model, optimizer, criterion, device=device, non_blocking=True)
    evaluator = create_supervised_evaluator(model,
                                            metrics={'eval_loss': Loss(get_loss_fn(train_cfg['loss_fn']))},
                                            device=device, non_blocking=True)

    # Configure Ignite objects
    evaluator.best_loss = 10000

    network_pass_timer = Timer(average=False)
    network_pass_timer.attach(trainer, start=Events.EPOCH_STARTED,
                              resume=Events.ITERATION_STARTED, pause=Events.ITERATION_COMPLETED)

    RunningAverage(output_transform=lambda x: x).attach(trainer, 'avg_loss')

    model_checkpoint_handler = ModelCheckpoint(save_dir, 'checkpoint', save_interval=train_cfg['save_model_interval'],
                                               n_saved=train_cfg['ignite_history_size'], require_empty=False)
    final_checkpoint_handler = ModelCheckpoint(save_dir, 'final', save_interval=1, n_saved=1, require_empty=False)
    best_model_handler = ModelCheckpoint(save_dir, 'best', save_interval=1, n_saved=1, require_empty=False)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, model_checkpoint_handler, {'model': model})
    trainer.add_event_handler(Events.COMPLETED, final_checkpoint_handler, {'model': model,
                                                                           'optimizer': optimizer})

    trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

    @trainer.on(Events.STARTED)
    def log_details_to_writer(engine: Engine):
        writer.add_text('config', str(cfg))
        writer.add_text('model', str(model))

        # loader = iter(val_loader)
        # x, y = next(loader)
        # try:
        #     writer.add_graph(model, input_to_model=x, verbose=True)
        # except Exception as e:
        #     print("Failed to save model graph: {}".format(e))

    @trainer.on(Events.EPOCH_STARTED)
    def step_scheduler(engine: Engine):
        if lr_scheduler is not None:
            lr_scheduler.step(engine.state.epoch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(train_engine: Engine):
        train_duration = timer_to_str(network_pass_timer.value())
        avg_loss = train_engine.state.metrics['avg_loss']
        msg = f'Training results - Epoch:{train_engine.state.epoch:2d}/{train_engine.state.max_epochs}. ' \
            f'Duration: {train_duration}. Avg loss: {avg_loss:.4f}'
        logger.info(msg)
        writer.add_scalar('training/avg_loss', avg_loss, train_engine.state.epoch)
        writer.add_scalar('timer/train_timer', network_pass_timer.value(), train_engine.state.epoch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_evaluation_results(train_engine: Engine):
        if data_cfg['run_val_on_train']:
            evaluator.run(val_train_loader)
            evaluation_loss = evaluator.state.metrics['eval_loss']
            msg = f'Eval. on val_train_loader - Epoch:{train_engine.state.epoch:2d}/{train_engine.state.max_epochs}. ' \
                f'Avg loss: {evaluation_loss:.4f}'
            logger.info(msg)
            writer.add_scalar('training_eval/avg_loss', evaluation_loss, train_engine.state.epoch)

        evaluator.run(val_loader)
        evaluation_loss = evaluator.state.metrics['eval_loss']
        msg = f'Eval. on val_loader - Epoch:{train_engine.state.epoch:2d}/{train_engine.state.max_epochs}. ' \
            f'Avg loss: {evaluation_loss:.4f}'
        logger.info(msg)
        writer.add_scalar('validation_eval/avg_loss', evaluation_loss, train_engine.state.epoch)

        if evaluation_loss < evaluator.best_loss:
            best_model_handler(train_engine, {'model': model, 'optimizer': optimizer})
            evaluator.best_loss = evaluation_loss

    @trainer.on(Events.EXCEPTION_RAISED)
    def handle_exception(train_engine: Engine, e):
        logger.info(e)
        if isinstance(e, KeyboardInterrupt):
            train_engine.terminate()
            model_checkpoint_handler(train_engine, {'model': model})
        raise e

    @trainer.on(Events.COMPLETED)
    def close_writer(engine: Engine):
        writer.export_scalars_to_json(os.path.join(save_dir, 'tensorboardX.json'))
        writer.close()

    # Run engine
    logger.info('All set. Start training!')
    trainer.run(train_loader, max_epochs=train_cfg['num_epochs'])


if __name__ == '__main__':
    # Load config file
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=CONFIG_STANDARD,
                        help='Configuration file to use')
    parser.add_argument('--ds_path', type=str, default=DATA_DIR_AT_AMC,
                        help='Path to main data directory')

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f)

    # Define data directory from args
    config['data']['path'] = args.ds_path

    # Create logger, writer
    logging_dir = get_new_run_path(config['run_name'])
    shutil.copy(args.config, logging_dir)

    train(config, logging_dir)
