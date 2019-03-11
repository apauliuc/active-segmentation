import shutil
import sys
import yaml
import argparse
from os.path import join

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import ModelCheckpoint, TerminateOnNan, Timer
from ignite.metrics import Loss, RunningAverage

from models import get_model
from losses import get_loss_fn
from definitions import DATA_DIR, CONFIG_STANDARD
from helpers.utils import create_logger
from helpers.types import device
from helpers.dataloader import create_data_loader
from helpers.paths import get_dataset_path, get_new_checkpoint_path


def train(cfg, save_dir):
    # Initialise writer, logger and configs
    writer = SummaryWriter(log_dir=save_dir)
    logger = create_logger(save_dir, __name__)

    data_cfg = cfg['data']
    model_cfg = cfg['model']
    train_cfg = cfg['training']

    # Create dataloaders
    train_path = get_dataset_path(data_cfg['path'], data_cfg['dataset'], data_cfg['train_split'])
    train_loader = create_data_loader(data_cfg, train_path)

    val_path = get_dataset_path(data_cfg['path'], data_cfg['dataset'], data_cfg['val_split'])
    val_loader = create_data_loader(data_cfg, val_path)

    logger.info('Data loaders created')

    # Create model, loss function and optimizer
    model = get_model(model_cfg,
                      train_loader.dataset.n_channels,
                      train_loader.dataset.n_classes).to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=train_cfg['lr'],
                           weight_decay=train_cfg['weight_decay'], amsgrad=train_cfg['amsgrad'])

    criterion = get_loss_fn(train_cfg['loss_fn']).to(device=device)

    logger.info(f'Using model {model}. Loss function: {criterion}')

    # Make resuming possible

    # Create engines
    trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
    evaluator = create_supervised_evaluator(model,
                                            metrics={'eval_loss': Loss(get_loss_fn(train_cfg['loss_fn']))},
                                            device=device)

    # Configure Ignite

    # epoch_timer = Timer(average=False)
    # epoch_timer.attach(trainer, start=Events.EPOCH_STARTED,
    #                    resume=Events.ITERATION_STARTED, pause=Events.ITERATION_COMPLETED)

    RunningAverage(output_transform=lambda x: x).attach(trainer, 'avg_loss')

    model_checkpoint_handler = ModelCheckpoint(save_dir, 'model', save_interval=train_cfg['save_model_interval'],
                                               n_saved=train_cfg['ignite_history_size'], require_empty=False)
    final_checkpoint_handler = ModelCheckpoint(save_dir, 'final', save_interval=1, n_saved=1, require_empty=False)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, model_checkpoint_handler, {'model': model})
    trainer.add_event_handler(Events.COMPLETED, final_checkpoint_handler, {'model': model})

    @trainer.on(Events.STARTED)
    def log_config_to_writer(engine: Engine):
        writer.add_text('config', config)

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine: Engine):
        iteration = (engine.state.iteration - 1) % len(train_loader) + 1
        if iteration % train_cfg['log_interval'] == 0:
            msg = f'Epoch[{engine.state.epoch}] Iteration[{iteration}/{len(train_loader)}]]'\
                  f'Loss: {engine.state.output}'
            logger.info(msg)
            # writer.add_scalar("training/loss", engine.state.output, engine.state.iteration)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine: Engine):
        avg_loss = engine.state.metrics['avg_loss']
        msg = f'Training results - Epoch:{engine.state.epoch:2d}/{engine.state.max_epochs}.'\
            f'Avg loss: {avg_loss:.4f}'
        logger.info(msg)
        writer.add_scalar("training/avg_loss", avg_loss, engine.state.epoch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_evaluation_results(engine: Engine):
        evaluator.run(train_loader)
        evaluation_loss = engine.state.metrics['eval_loss']
        msg = f'Eval. on train_loader - Epoch:{engine.state.epoch:2d}/{engine.state.max_epochs}.' \
            f'Avg loss: {evaluation_loss:.4f}'
        logger.info(msg)
        writer.add_scalar("training_eval/avg_loss", evaluation_loss, engine.state.epoch)

        evaluator.run(val_loader)
        evaluation_loss = engine.state.metrics['eval_loss']
        msg = f'Eval. on val_loader - Epoch:{engine.state.epoch:2d}/{engine.state.max_epochs}.'\
              f'Avg loss: {evaluation_loss:.4f}'
        logger.info(msg)
        writer.add_scalar("validation_eval/avg_loss", evaluation_loss, engine.state.epoch)

    @trainer.on(Events.EXCEPTION_RAISED)
    def handle_exception(engine: Engine, e):
        if isinstance(e, KeyboardInterrupt):
            engine.terminate()
            model_checkpoint_handler(engine, {'model': model})
        raise e

    @trainer.on(Events.COMPLETED)
    def close_writer(engine: Engine):
        writer.export_scalars_to_json(join(save_dir, 'tensorboardX.json'))
        writer.close()

    # Run engine
    logger.info('All set. Start training!')
    trainer.run(train_loader, train_cfg['num_epochs'])


if __name__ == '__main__':
    # Load config file
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=CONFIG_STANDARD,
                        help='Configuration file to use')
    parser.add_argument('--ds_path', type=str, default=DATA_DIR,
                        help='Path to main data directory')

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f)

    # Define data directory from args
    config['data']['path'] = args.ds_path

    # Create logger, writer
    logging_dir = get_new_checkpoint_path()
    shutil.copy(args.config, logging_dir)

    train(config, logging_dir)
