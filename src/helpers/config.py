import yaml
from os.path import join

from definitions import CONFIG_DIR, CONFIG_DEFAULT


class ConfigClass(object):
    """
    @DynamicAttrs
    """

    def __init__(self, dictionary):
        self.update(dictionary)

    def __repr__(self):
        return str(self.__dict__)

    @staticmethod
    def _as_config_class(key, element):
        if isinstance(element, dict):
            return key, ConfigClass(element)
        else:
            return key, element

    @staticmethod
    def _as_dict(element):
        if isinstance(element, ConfigClass):
            return element.state_dict()
        else:
            return element

    @staticmethod
    def _update(key, dictionary, dictionary_new):
        if dictionary is None:
            return key, dictionary_new

        if isinstance(dictionary, dict):
            if isinstance(dictionary_new, dict):
                dictionary.update(dict(ConfigClass._update(k, dictionary.get(k), dictionary_new.get(k))
                                       for k, v in dictionary_new.items()))
                return key, dictionary
            else:
                return key, dictionary_new
        else:
            return key, dictionary_new

    def update(self, dictionary_new=None):
        dictionary_new = {} if dictionary_new is None else dictionary_new

        if isinstance(dictionary_new, ConfigClass):
            dictionary_new = dictionary_new.state_dict()

        dictionary = self.state_dict()
        dictionary.update(dict(self._update(k, dictionary.get(k), dictionary_new.get(k))
                               for k, v in dictionary_new.items()))
        self.__dict__.update(dictionary)

        obj_dictionary = dict(self._as_config_class(k, v) for k, v in self.__dict__.items())
        self.__dict__.update(obj_dictionary)

        return self

    def items(self):
        return self.__dict__.items()

    def state_dict(self):
        return dict((k, self._as_dict(v)) for k, v in self.__dict__.items())


def get_default_config() -> dict:
    with open(join(CONFIG_DIR, CONFIG_DEFAULT), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        assert config is not None
        return config


def get_config_from_path(config_path: str) -> ConfigClass:
    with open(config_path, 'r') as f_new:
        default_config = get_default_config()
        custom_config = yaml.load(f_new, Loader=yaml.FullLoader)

        config = ConfigClass(default_config)
        config.update(custom_config)

        return config
