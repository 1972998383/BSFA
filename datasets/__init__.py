from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .CUB import CUB_200_2011
from .Dogs import StanfordDogs
from .Cars import StanfordCars
from .Leaf import Leaf
from .Cell import Cell
from .Animal import Animal
from .FGSCR import FGSCR

__imgfewshot_factory = {
        'CUB_200_2011': CUB_200_2011,
        'StanfordDogs': StanfordDogs,
        'StanfordCars': StanfordCars,
        'Leaf': Leaf,
        'Cell': Cell,
        'Animal': Animal,
        'FGSCR': FGSCR
}


def get_names():
    return list(__imgfewshot_factory.keys()) 


def init_imgfewshot_dataset(name, **kwargs):
    if name not in list(__imgfewshot_factory.keys()):
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(name, list(__imgfewshot_factory.keys())))
    return __imgfewshot_factory[name](**kwargs)

