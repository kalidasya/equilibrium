import copy
from functools import partial

import pygame
from deap import gp

from creatures import base
from utils import func


class Bacteria(base.Individual):

    def __init__(self, world, config, tree=None, center=None, individuals=None):
        super().__init__(world, config, tree=tree, center=center)
        self.individuals = individuals

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

    def get_color(self):
        return 255 - (100 - int(self.energy)), 0, 0, 255

    def eval(self):
        return self.energy

    def move(self):
        super().move()
        food = pygame.sprite.spritecollideany(self, self.individuals)
        if food and not isinstance(food, Bacteria):
            food.kill()
            self.energy += 100

    def is_food_ahead(self):
        tmp_sprite = pygame.sprite.Sprite()
        tmp_sprite.rect = copy.deepcopy(self.rect)
        tmp_sprite.rect.move_ip(self._get_move_vector())
        food = pygame.sprite.spritecollideany(tmp_sprite, self.individuals)
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
        ret.individuals = self.individuals
        return ret

    def __repr__(self):
        return f"<Bacteria {id(self)}>"


class BacteriaGroup(base.Population):
    pass
