import copy
from functools import partial

import pygame
from deap import gp

from creatures import base
from utils import func


class Bacteria(base.Individual):

    def __init__(self, world, config, tree=None, center=None, sprites=None):
        super().__init__(world, config, tree=tree, center=center)
        self.sprites = sprites

    @classmethod
    def set_brain(cls, bacteria):
        pset = gp.PrimitiveSet("BACTERIA", 0)
        pset.addPrimitive(bacteria.if_food_ahead, 2)
        pset.addPrimitive(func.prog2, 2)
        pset.addPrimitive(func.prog3, 3)
        pset.addTerminal(bacteria.move)
        pset.addTerminal(bacteria.rotate_left)
        pset.addTerminal(bacteria.rotate_right)
        return pset

    def update_color(self):
        self.image.fill((255 - (100 - self.energy), 0, 0, 255))

    def eval(self):
        return self.energy

    def move(self):
        super().move()
        food = pygame.sprite.spritecollideany(self, self.sprites)
        if food and not isinstance(food, Bacteria):
            food.kill()
            self.energy += 100

    def is_food_ahead(self):
        rect_tmp = copy.deepcopy(self.rect)
        self.rect.move_ip(self._get_move_vector())
        food = pygame.sprite.spritecollideany(self, self.sprites)
        self.rect = rect_tmp
        return bool(food)

    def if_food_ahead(self, out1, out2):
        return partial(func.if_then_else, self.is_food_ahead, out1, out2)

    def can_mate(self):
        """
        :return:
        """
        return not self.mated

    def copy(self):
        ret = super().copy()
        ret.sprites = self.sprites
        return ret

    def __repr__(self):
        return f"<Bacteria {id(self)}>"


class BacteriaGroup(base.Population):
    pass
