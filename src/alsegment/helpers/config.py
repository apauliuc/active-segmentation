class ConfigClass(object):
    """
    @DynamicAttrs
    """

    @staticmethod
    def _traverse(key, element):
        if isinstance(element, dict):
            return key, ConfigClass(element)
        else:
            return key, element

    def __init__(self, dictionary):
        obj_dictionary = dict(self._traverse(k, v) for k, v in dictionary.items())
        self.__dict__.update(obj_dictionary)

    def update(self, dictionary):
        obj_dictionary = dict(self._traverse(k, v) for k, v in dictionary.items())
        self.__dict__.update(obj_dictionary)

    def __contains__(self, key):
        return True

    def items(self):
        return self.__dict__.items()
