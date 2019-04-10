from torch.utils.data import DataLoader


class BaseLoader:
    train_loader: DataLoader
    val_loader: DataLoader
    val_train_loader: DataLoader
    msg: str
    input_channels: int
    num_classes: int
