import copy
import dataclasses
import operator
import random

import pygraphviz as pgv
import pygame.sprite
from deap import gp
from deap.gp import PrimitiveTree
from deap.tools import selection

from utils.ranged_number import RangedNumber

bloat_limiter = gp.staticLimit(operator.attrgetter('height'), 17)
limited_crossover = bloat_limiter(gp.cxOnePoint)
limited_mutation = bloat_limiter(gp.mutUniform)


@dataclasses.dataclass
class IndividualConfig:
    width: int
    height: int
    rotate_drain: int
    move_drain: int
    photosynthesis_gain: int
    mating_percent: float
    mutation_chance: float
    crowded_threshold: int
    reset_percent: float
    dead: int
    mating_rect_width: int
    mating_rect_height: int


class Individual(pygame.sprite.Sprite):
    def __init__(self, world: pygame.Surface, config: IndividualConfig, tree=None, center=None):
        super().__init__()

        self.world = world
        self.config = config
        self.fissioned = False
        self.mated = False
        self.screen = self.world.get_size()
        self.image = pygame.Surface((self.config.width, self.config.height))
        self.image.fill((0, 255, 0))
        self.rect = self.image.get_rect()
        if center:
            self.rect.center = center
        else:
            self.rect.center = (random.randint(self.config.width, self.screen[0] - self.config.width),
                                random.randint(self.config.height, self.screen[1] - self.config.height))
        self.mating_rect = self.rect.inflate(self.config.mating_rect_width, self.config.mating_rect_height)

        self.energy = RangedNumber(0, 100, 60)
        self.dir = random.randint(0, 3)

        self.pset = self.set_brain(self)
        if tree is None:
            self.tree = PrimitiveTree(self.expr_init())
        else:
            self.tree = copy.deepcopy(tree)

    def copy(self):
        ret = self.__class__(world=self.world, config=self.config, tree=self.tree, center=self.rect.center)
        ret.energy = self.energy
        return ret

    @classmethod
    def mate(cls, ind1: 'Individual', ind2: 'Individual'):
        ind1.mated, ind2.mated = True, True
        tree1, tree2 = limited_crossover(copy.deepcopy(ind1.tree), copy.deepcopy(ind2.tree))
        child1 = ind1.copy()
        child1.tree = tree1

        child2 = ind2.copy()
        child2.tree = tree1
        return child1, child2

    def expr_init(self, pset=None, type_=None):
        if pset is None:
            return gp.genFull(self.pset, min_=1, max_=2, type_=type_)
        return gp.genFull(pset, min_=1, max_=2, type_=type_)

    def mutate(self):
        tree = limited_mutation(self.tree, expr=self.expr_init, pset=self.pset)
        self.tree = tree[0]

    def rotate_left(self):
        self.energy -= self.config.rotate_drain
        self.dir = (self.dir - 1) % 4

    def rotate_right(self):
        self.energy -= self.config.rotate_drain
        self.dir = (self.dir + 1) % 4

    def _get_move_vector(self):
        current_pos = self.rect.center
        new_vector = [0, 0]
        match self.dir:
            case 0 if current_pos[1] > self.config.height:
                new_vector[1] -= self.config.height
            case 1 if current_pos[0] < self.screen[0] - self.config.width:
                new_vector[0] += self.config.width
            case 2 if current_pos[1] < self.screen[1] - self.config.height:
                new_vector[1] += self.config.height
            case 3 if current_pos[0] > self.config.width:
                new_vector[0] -= self.config.width
        return new_vector

    def move(self):
        self.energy -= self.config.move_drain
        move = self._get_move_vector()
        self.rect.move_ip(*move)
        self.mating_rect.move_ip(*move)

    def run(self, routine):
        routine()

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


def eval_individual(tree, pset, entity):
    # Transform the tree expression to functional Python code
    routine = gp.compile(tree, pset)
    # Run the generated routine
    entity.run(routine)
    return entity.eval()


def vicinity_collision(left, right):
    if left != right:
        return left.mating_rect.colliderect(right.rect)
    else:
        return False


class Population(pygame.sprite.Group):

    def __init__(self, config):
        super().__init__()
        self.config = config

    def eval(self):
        """
        Evaluate and update all entities
        :return:
        """
        for ind in self.sprites():
            eval_individual(ind.tree, ind.pset, ind)
            ind.update_color()

    def grow_population(self):
        """
                Mate the population aka create new entities if:
                There are entities which are in close proximity
                Algae.can_mate returned true
                :return:
                """
        children = []
        if len(self.sprites()) > 1:
            possible_mates = pygame.sprite.groupcollide(
                self.sprites(), self.sprites(), False, False,
                collided=vicinity_collision)

            mating_percent = self.config.mating_percent
            possible_mates_sorted = sorted([i for i in possible_mates.keys() if i.can_mate()], key=lambda x: x.eval(), reverse=True)
            possible_mates_sorted = possible_mates_sorted[:int(len(possible_mates_sorted) * mating_percent)]

            for base in possible_mates_sorted:
                mates = sorted([p for p in possible_mates[base] if not p.mated], key=lambda x: x.eval(), reverse=True)
                if len(mates) < base.config.crowded_threshold:
                    # print(len(mates))
                    for partner in mates:
                        if partner.can_mate():
                            # print("mating")
                            offsprings = base.mate(base, partner)
                            children.extend(offsprings)
                            self.add(offsprings)

        return children

    def reduce_population(self):
        """
        Reduce population with the rule: entity evals to entity.DEAD considered dead
        :return:
        """
        dead_entities = [a for a in self.sprites() if a.eval() == a.config.dead]
        for entity in dead_entities:
            entity.kill()

    def reset_mated(self):
        """
        Reset k% of the mated entities to ready to be mated again
        :param k:
        :return:
        """
        # reset mated for .1 of the mated
        mated = [a for a in self.sprites() if a.mated]
        for ind in selection.selRandom(mated, k=int(len(mated) * self.config.reset_percent)):
            ind.mated = False

    def mutate_population(self):
        for ind in self.sprites():
            if random.random() <= ind.config.mutation_chance:
                ind.mutate()
