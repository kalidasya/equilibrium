import pygraphviz as pgv
import random
from functools import partial

import numpy as np
import pygame
from deap import gp, creator, base
from deap.gp import PrimitiveTree

from utils import func
from utils.ranged_number import RangedNumber

creator.create("FitnessMax", base.Fitness, weights=(1.0,1.0))


def set_brain(algae):
    pset = gp.PrimitiveSet("ALGAE", 0)
    pset.addPrimitive(algae.if_lighter_ahead, 2)
    pset.addPrimitive(func.prog2, 2)
    pset.addPrimitive(func.prog3, 3)
    pset.addTerminal(algae.move)
    pset.addTerminal(algae.rotate_left)
    pset.addTerminal(algae.rotate_right)
    return pset


def eval_algae(tree, pset, algae):
    # Transform the tree expression to functionnal Python code
    routine = gp.compile(tree, pset)
    # Run the generated routine
    algae.run(routine)
    return algae.eval()


class Algae(pygame.sprite.Sprite):
    WIDTH = 2
    HEIGHT = 2
    WATER_LIMIT = 1
    # WATER_LIMIT = 0.8
    LIGHT_LIMIT = 1
    # LIGHT_LIMIT = 0.8

    fitness = creator.FitnessMax

    def __init__(self, world: pygame.Surface, copy=False):
        super().__init__()
        self.world = world
        self.screen = self.world.get_size()
        if not copy:
            self.image = pygame.Surface((self.WIDTH, self.HEIGHT))
            self.image.fill((0, 255, 0))
            self.rect = self.image.get_rect()
            self.rect.center = (random.randint(self.WIDTH, self.screen[0] - self.WIDTH), random.randint(self.HEIGHT, self.screen[1] - self.HEIGHT))

            self._dryness = RangedNumber(0, 100, 0)
            self._paleness = RangedNumber(0, 100, 0)
            self._age = RangedNumber(0, 100, 0)
            self.energy = RangedNumber(0, 100, 100)
            # self.fitness = 1
            self.dir = random.randint(0, 3)
            self.pset = set_brain(self)
            self.tree = PrimitiveTree(self.expr_init())

    def expr_init(self, pset=None, type_=None):
        if pset is None:
            return gp.genFull(self.pset, min_=1, max_=2, type_=type_)
        return gp.genFull(pset, min_=1, max_=2, type_=type_)

    def mutate(self):
        return gp.mutUniform(self.tree, expr=self.expr_init, pset=self.pset)[0]

    def age(self):
        self._age += 1

    def _get_light_level(self, coords=None):
        if coords is None:
            sun, *_ = self.world.get_at(self.rect.center)
        else:
            sun, *_ = self.world.get_at(coords)
        return sun / 255

    def _get_water_level(self):
        _, _, water, _ = self.world.get_at(self.rect.center)
        return water / 255

    @property
    def dryness(self):
        water_level = self._get_water_level()
        self._dryness = 0
        if water_level < self.WATER_LIMIT:
            self._dryness = water_level / self.WATER_LIMIT

        return self._dryness
        # print(f"{light_level=} {water_level=}")

        # if water_level < self.WATER_LIMIT:
        #     self._dryness += 10 * (1 - water_level)
        # else:
        #     self._dryness -= 10 * (1 - water_level)

    @property
    def paleness(self):
        light_level = self._get_light_level()
        self._paleness = 0
        if light_level < self.WATER_LIMIT:
            self._paleness = light_level / self.WATER_LIMIT
        return self._paleness

        # if light_level < self.LIGHT_LIMIT:
        #     self.paleness += 10 * (1 - light_level)
        # else:
        #     self.paleness -= 10 * (1 - light_level)

    def update_color(self):
        self.image.fill((0, 255 - (int(self.dryness) + int(self.paleness)), 0, 255))

    def eval(self):
        return 100 - self.dryness, 100 - self.paleness

    def rotate_left(self):
        self.dir = (self.dir - 1) % 4

    def rotate_right(self):
        self.dir = (self.dir + 1) % 4

    def _get_move_vector(self):
        current_pos = self.rect.center
        new_vector = [0, 0]
        match self.dir:
            case 0 if current_pos[1] > self.HEIGHT:
                new_vector[1] -= self.HEIGHT
            case 1 if current_pos[0] < self.screen[0] - self.WIDTH:
                new_vector[0] += self.WIDTH
            case 2 if current_pos[1] < self.screen[1] - self.HEIGHT:
                new_vector[1] += self.HEIGHT
            case 3 if current_pos[0] > self.WIDTH:
                new_vector[0] -= self.WIDTH
        return new_vector

    def move(self):
        self.rect.move_ip(*self._get_move_vector())

    def lighter_ahead(self):
        target_coords = np.add(self.rect.center, self._get_move_vector())
        target_light_level = self._get_light_level(target_coords)
        current_light_level = self._get_light_level()
        # print(f"looking at: {target_coords} {target_light_level=} {current_light_level=}")
        return target_light_level >= current_light_level

    def if_lighter_ahead(self, out1, out2):
        return partial(func.if_then_else, self.lighter_ahead, out1, out2)

    def run(self, routine):
        routine()

    def copy(self):
        copyobj = Algae(self.world)
        copyobj.pset = self.pset
        copyobj.tree = self.tree
        return copyobj

    def save_tree(self):
        nodes, edges, labels = gp.graph(self.tree)

        g = pgv.AGraph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        g.layout(prog="dot")

        for i in nodes:
            n = g.get_node(i)
            n.attr["label"] = labels[i]

        g.draw("tree.pdf")

