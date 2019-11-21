from torch.utils.data import Dataset, DataLoader


class BaseLoader:
    train_loader: DataLoader
    train_dataset: Dataset
    val_loader: DataLoader
    val_dataset: Dataset
    val_train_loader: DataLoader
    msg: str
    input_channels: int
    num_classes: int
    predict_path: str
    dir_list: list
    image_size: tuple
    ds_statistics: dict

    def update_train_loader(self, new_file_list: list) -> None:
        raise NotImplementedError
