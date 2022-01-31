import copy
import dataclasses
from functools import partial

import pygame
from deap import gp

from creatures import base
from creatures.base import IndividualConfig, IndividualMenu
from utils import func


def food_sensor_collision(left, right):
    if left != right:
        return left.food_sensor_rect.colliderect(right.rect)
    else:
        return False


@dataclasses.dataclass
class BacteriaConfig(IndividualConfig):
    food_sensing_distance: int


class BacteriaMenu(IndividualMenu):

    def _draw(self, pos):
        pos += self._input2("Food sensing distance", pygame.Rect((0, pos), (self.width, self.row_height)), "food_sensing_distance")



class Bacteria(base.Individual):
    name = "Bacteria"

    def __init__(self, world, config, tree=None, center=None, individuals=None):
        super().__init__(world, config, tree=tree, center=center)
        self.individuals = individuals
        self.food_sensor_rect = copy.deepcopy(self.rect)
        self.update_food_sensor_rect()

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

    def move(self):
        vector = super().move()
        self.food_sensor_rect.move_ip(*vector)
        foods = pygame.sprite.spritecollide(self, self.individuals, False)
        for food in foods:
            if food and not isinstance(food, Bacteria):
                self._inc_energy(100)
                break

    def rotate_right(self):
        super(Bacteria, self).rotate_right()
        self.update_food_sensor_rect()

    def rotate_left(self):
        super(Bacteria, self).rotate_left()
        self.update_food_sensor_rect()

    def update_food_sensor_rect(self):
        match self.dir:
            case 0:
                self.food_sensor_rect.width = self.rect.width
                self.food_sensor_rect.height = self.config.food_sensing_distance
                self.food_sensor_rect.center = self.rect.centerx, self.rect.centery - self.config.food_sensing_distance // 2
            case 1:
                self.food_sensor_rect.width = self.config.food_sensing_distance
                self.food_sensor_rect.height = self.rect.height
                self.food_sensor_rect.center = self.rect.centerx - self.config.food_sensing_distance // 2, self.rect.centery
            case 2:
                self.food_sensor_rect.width = self.rect.width
                self.food_sensor_rect.height = self.config.food_sensing_distance
                self.food_sensor_rect.center = self.rect.centerx, self.rect.centery + self.config.food_sensing_distance // 2
            case 3:
                self.food_sensor_rect.width = self.config.food_sensing_distance
                self.food_sensor_rect.height = self.rect.height
                self.food_sensor_rect.center = self.rect.centerx + self.config.food_sensing_distance // 2, self.rect.centery

    def is_food_ahead(self):
        # TODO almost copy the whole thing first...
        tmp_sprite = pygame.sprite.Sprite()
        tmp_sprite.rect = copy.deepcopy(self.rect)
        tmp_sprite.food_sensor_rect = copy.deepcopy(self.food_sensor_rect)
        vector = self._get_move_vector()
        tmp_sprite.rect.move_ip(*vector)
        tmp_sprite.food_sensor_rect.move_ip(*vector)

        food = pygame.sprite.spritecollide(tmp_sprite, self.individuals, False, collided=food_sensor_collision)
        return any([not isinstance(f, Bacteria) for f in food])

    def if_food_ahead(self, out1, out2):
        return partial(func.if_then_else, self.is_food_ahead, out1, out2)

    def copy(self):
        ret = super().copy()
        ret.individuals = self.individuals
        return ret

    def __repr__(self):
        return f"<Bacteria {id(self)} {self.rect}>"


class BacteriaGroup(base.Population):
    pass
    # def as_group(self):
    #     ret = pygame.sprite.Group()
    #     for ind in self:
    #         s = pygame.sprite.Sprite()
    #         s.image = pygame.Surface((ind.food_sensor_rect.width, ind.food_sensor_rect.height))
    #         s.image.fill(ind.get_color())
    #         s.rect = ind.food_sensor_rect
    #         ret.add(s)
    #     return ret
