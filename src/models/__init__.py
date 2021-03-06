from models.unet import UNet
from models.ternausnet import TernausNet
from models.segnet import SegNet
from models.fcn8 import FCN8
from models.fcn16 import FCN16
from models.fcn32 import FCN32
from bayesian.bayes_unet import BBBUnet
from models.unet_git import UNetOtherVersion
from models.unet_proba import ProbabilisticUNet
from models.unet_proba_spatial_common import ProbaUNetSpatialCommon
from models.unet_proba_spatial import ProbabilisticUNetSpatial
from models.unet_proba_spatial_previous import ProbabilisticUNetSpatialPrevious
from models.unet_pytorch import UNetPyTorch
from models.unet_sk import SKUNet
from models.unet_vae import VariationalUNet
from models.unet_vae_no_recon import VariationalUNetNoRecon
from models.unet_vae_stochastic import StochasticUNetNoRecon

from helpers.utils import retrieve_class_init_parameters


def get_model(model_cfg):
    model_name = model_cfg.name
    model_cls = _get_model_instance(model_name, model_cfg.type)

    init_param_names = retrieve_class_init_parameters(model_cls)
    param_dict = {k: v for k, v in model_cfg.network_params.items() if k in init_param_names}

    model = model_cls(**param_dict)

    return model


def _get_model_instance(name: str, train_type: str):
    try:
        if train_type == 'standard':
            return {
                'unet_git': UNetOtherVersion,
                'unet_pytorch': UNetPyTorch,
                'unet': UNet,
                'skunet': SKUNet,
                'ternaus_net': TernausNet,
                'segnet': SegNet,
                'fcn8': FCN8,
                'fcn16': FCN16,
                'fcn32': FCN32
            }[name]
        elif train_type == 'bayesian':
            return BBBUnet
        elif train_type == 'variational':
            return {
                'unet_proba': ProbabilisticUNet,
                'unet_proba_spatial': ProbabilisticUNetSpatial,
                'unet_proba_common': ProbaUNetSpatialCommon,
                'unet_vae': VariationalUNet,
                'unet_vae_no_recon': VariationalUNetNoRecon,
                'unet_proba_spatial_prev': ProbabilisticUNetSpatialPrevious,
                'unet_stochastic_no_recon': StochasticUNetNoRecon
            }[name]
    except KeyError:
        raise Exception(f'Model {name} not available')
