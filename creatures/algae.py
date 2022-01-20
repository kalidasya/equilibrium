import copy

import pygraphviz as pgv
import random
from functools import partial

import numpy as np
import pygame
from deap import gp, creator, base
from deap.gp import PrimitiveTree
from deap.tools import selection

from utils import func
from utils.ranged_number import RangedNumber

# TODO this needs to go
creator.create("FitnessMax", base.Fitness, weights=(1.0,))


def set_brain(algae):
    pset = gp.PrimitiveSet("ALGAE", 0)
    pset.addPrimitive(algae.if_lighter_ahead, 2)
    pset.addPrimitive(algae.if_wetter_ahead, 2)
    pset.addPrimitive(func.prog2, 2)
    pset.addPrimitive(func.prog3, 3)
    pset.addTerminal(algae.move)
    pset.addTerminal(algae.photosynthesize)
    pset.addTerminal(algae.rotate_left)
    pset.addTerminal(algae.rotate_right)
    return pset


def eval_algae(tree, pset, algae):
    # Transform the tree expression to functional Python code
    routine = gp.compile(tree, pset)
    # Run the generated routine
    algae.run(routine)
    return algae.eval()


class Algae(pygame.sprite.Sprite):
    WIDTH = 2
    HEIGHT = 2
    ROTATE_DRAIN = 1
    MOVE_DRAIN = 2
    PHOTOSYNTHESIS_GAIN = 100
    # MATING_DRAIN = 20
    # MATING_LIMIT = 50,
    MATING_CHANCE = .1
    MUTATION_CHANCE = .1
    CROWDED_THRESHOLD = 3
    DEAD = 0,

    fitness = creator.FitnessMax

    def __init__(self, world: pygame.Surface, tree=None, center=None):
        super().__init__()
        self.world = world
        self.screen = self.world.get_size()
        self.image = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.image.fill((0, 255, 0))
        self.rect = self.image.get_rect()
        if center:
            self.rect.center = center
        else:
            self.rect.center = (random.randint(self.WIDTH, self.screen[0] - self.WIDTH),
                                random.randint(self.HEIGHT, self.screen[1] - self.HEIGHT))
        self.mating_rect = self.rect.inflate(6, 6)
        self.mated = False

        self.energy = RangedNumber(0, 100, 60)
        self.dir = random.randint(0, 3)

        self.pset = set_brain(self)
        if tree is None:
            self.tree = PrimitiveTree(self.expr_init())
        else:
            self.tree = copy.deepcopy(tree)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['world'] = None
        state['screen'] = None
        state['image'] = None
        return state

    def expr_init(self, pset=None, type_=None):
        if pset is None:
            return gp.genFull(self.pset, min_=1, max_=2, type_=type_)
        return gp.genFull(pset, min_=1, max_=2, type_=type_)

    def mutate(self):
        tree = gp.mutUniform(self.tree, expr=self.expr_init, pset=self.pset)
        self.tree = tree[0]

    @classmethod
    def mate(cls, algae1: 'Algae', algae2: 'Algae'):
        algae1.mated, algae2.mated = True, True
        children = gp.cxOnePoint(copy.deepcopy(algae1.tree), copy.deepcopy(algae2.tree))
        choice = random.randint(0, 1)
        ret = Algae(algae1.world, tree=children[choice], center=algae1.rect.center)
        ret.mated = True  # avoid immediate mating
        return ret

    def _get_light_level(self, coords=None):
        if coords is None:
            sun, *_ = self.world.get_at(self.rect.center)
        else:
            sun, *_ = self.world.get_at(coords)
        return sun / 255

    def _get_water_level(self, coords=None):
        if coords is None:
            _, _, water, _ = self.world.get_at(self.rect.center)
        else:
            _, _, water, _ = self.world.get_at(coords)
        return water / 255

    def update_color(self):
        self.image.fill((0, 255 - (100 - self.energy), 0, 255))

    def eval(self):
        return self.energy,

    def rotate_left(self):
        self.energy -= self.ROTATE_DRAIN
        self.dir = (self.dir - 1) % 4

    def rotate_right(self):
        self.energy -= self.ROTATE_DRAIN
        self.dir = (self.dir + 1) % 4

    def photosynthesize(self):
        efficiency = min(self._get_light_level(), self._get_water_level())
        # print(f"Gaining {int(self.PHOTOSYNTHESIS_GAIN * efficiency)} energy with {efficiency=}")
        self.energy += int(self.PHOTOSYNTHESIS_GAIN * efficiency)

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
        self.energy -= self.MOVE_DRAIN
        move = self._get_move_vector()
        self.rect.move_ip(*move)
        self.mating_rect.move_ip(*move)

    def lighter_ahead(self):
        target_coords = np.add(self.rect.center, self._get_move_vector())
        target_light_level = self._get_light_level(target_coords)
        current_light_level = self._get_light_level()
        # print(f"looking at: {target_coords} {target_light_level=} {current_light_level=}")
        return target_light_level >= current_light_level

    def wetter_ahead(self):
        target_coords = np.add(self.rect.center, self._get_move_vector())
        target_water_level = self._get_water_level(target_coords)
        current_water_level = self._get_water_level()
        # print(f"looking at: {target_coords} {target_light_level=} {current_light_level=}")
        return target_water_level >= current_water_level

    def if_lighter_ahead(self, out1, out2):
        return partial(func.if_then_else, self.lighter_ahead, out1, out2)

    def if_wetter_ahead(self, out1, out2):
        return partial(func.if_then_else, self.wetter_ahead, out1, out2)

    def can_mate(self, mates):
        """
        only mate if: mated flag is false
        AND the possible mates are less than the crowded threshold (?)
        AND the mating chance is hit
        :param mates:
        :return:
        """
        return not self.mated and 0 < len(mates) <= self.CROWDED_THRESHOLD and random.random() <= self.MATING_CHANCE

    def run(self, routine):
        routine()

    def copy(self):
        copyobj = Algae(self.world)
        copyobj.energy = self.energy
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

    def __repr__(self):
        return f"<Algae {id(self)}>"


def vicinity_collision(left, right):
    if left != right:
        return left.mating_rect.colliderect(right.rect)
    else:
        return False


class AlgaeGroup(pygame.sprite.Group):

    def eval_population(self):
        """
        Evaluate and update all entities
        :return:
        """
        for entity in self.sprites():
            res = eval_algae(entity.tree, entity.pset, entity)
            entity.fitness.values = res
            entity.update_color()

    def reduce_population(self):
        """
        Reduce population with the rule: entity evals to Algae.DEAD considered dead
        :return:
        """
        dead_algae = [a for a in self.sprites() if a.eval() == Algae.DEAD]
        for algae in dead_algae:
            algae.kill()

    def mate_population(self):
        """
        Mate the population aka create new entities if:
        There are entities which are in close proximity
        Algae.can_mate returned true
        :return:
        """
        children = []
        possible_mates = pygame.sprite.groupcollide(
            self.sprites(), self.sprites(), False, False,
            collided=vicinity_collision)

        possible_mates_sorted = sorted(list(possible_mates.keys()), key=lambda x: x.eval(), reverse=True)
        for base in possible_mates_sorted:
            mates = sorted([p for p in possible_mates[base] if not p.mated], key=lambda x: x.eval(), reverse=True)
            # print(len(mates))
            if base.can_mate(mates):
                # print("mating")
                child = Algae.mate(base, mates[0])
                children.append(child)
                self.add(child)

        return children

    def reset_mated(self, k):
        """
        Reset k% of the mated entities to ready to be mated again
        :param k:
        :return:
        """
        # reset mated for .1 of the mated
        ready_to_mate = [a for a in self.sprites() if a.mated]
        for algae in selection.selRandom(ready_to_mate, k=len(ready_to_mate) // k):
            algae.mated = False