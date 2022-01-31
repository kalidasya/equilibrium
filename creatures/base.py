import copy
import dataclasses
import multiprocessing
import operator
import random
from multiprocessing.pool import ThreadPool

import numpy as np
import pygame_gui
import pygraphviz as pgv
import pygame.sprite
from deap import gp
from deap.gp import PrimitiveTree
from deap.tools import selection
from pygame_gui.core import ObjectID

bloat_limiter = gp.staticLimit(operator.attrgetter('height'), 17)
limited_crossover = bloat_limiter(gp.cxOnePoint)
limited_mutation = bloat_limiter(gp.mutUniform)


@dataclasses.dataclass
class IndividualConfig:
    width: int
    height: int
    rotate_drain: int
    move_drain: int
    mating_percent: float
    mutation_chance: float
    crowded_threshold: int
    population_limit: int
    reset_percent: float
    dead: int
    mating_rect_width: int
    mating_rect_height: int
    hayflick_limit: int
    max_energy: int
    color: tuple
    # color: tuple


@dataclasses.dataclass
class GameConfig:
    algae_population: int
    bacteria_population: int
    generation: int
    game_speed: int
    paused: bool


@dataclasses.dataclass
class WorldResources:
    water: np.array
    light: np.array
    width: int
    height: int


class BottomPanel:
    def __init__(self, label, config: GameConfig, manager, menu_pos):
        self.manager = manager
        self.panel = pygame_gui.elements.UIPanel(
            starting_layer_height=1,
            relative_rect=menu_pos,
            manager=manager)
        self.config = config
        self.width = self.panel.rect.width
        self.row_height = 25
        self.algae_count = None
        self.bacteria_count = None
        self.generation_count = None
        self.inputs = {}

    def draw(self):
        rect = pygame.Rect((0, 0), (self.width // 4, self.row_height))
        pygame_gui.elements.UILabel(
            relative_rect=rect,
            text="Algae count:",
            manager=self.manager,
            container=self.panel,
        )
        self.algae_count = pygame_gui.elements.UILabel(
            relative_rect=rect.move(self.width // 4, 0),
            text=str(self.config.algae_population),
            manager=self.manager,
            container=self.panel,
        )
        pygame_gui.elements.UILabel(
            relative_rect=rect.move(self.width // 4 * 2, 0),
            text="Bacteria count:",
            manager=self.manager,
            container=self.panel,
        )
        self.bacteria_count = pygame_gui.elements.UILabel(
            relative_rect=rect.move(self.width // 4 * 3, 0),
            text=str(self.config.bacteria_population),
            manager=self.manager,
            container=self.panel,
        )

        rect = pygame.Rect((0, self.row_height), (self.width // 3, self.row_height))
        p = pygame_gui.elements.UILabel(
            relative_rect=rect,
            text="Gen count:",
            manager=self.manager,
            container=self.panel,
        )
        p.text_horiz_alignment = 'left'

        self.generation_count = pygame_gui.elements.UILabel(
            relative_rect=rect.move(self.width // 3, 0),
            text=str(self.config.generation),
            manager=self.manager,
            container=self.panel,
        )
        self.generation_count.text_horiz_alignment = 'left'

        # rect = pygame.Rect((0, self.row_height * 2), (self.width // 3, self.row_height))
        # p = pygame_gui.elements.UILabel(
        #     relative_rect=rect,
        #     text="Game speed:",
        #     manager=self.manager,
        #     container=self.panel,
        # )
        # p.text_horiz_alignment = 'left'
        # game_speed = pygame_gui.elements.UIHorizontalSlider(
        #     relative_rect=rect.move(self.width // 3, 0),
        #     start_value=self.config.game_speed,
        #     value_range=range(1, 60),
        #     manager=self.manager,
        #     container=self.panel,
        # )
        # game_speed.i_type = int
        # self.inputs[game_speed] = "game_speed"

        rect = pygame.Rect((self.width // 3 * 2, self.row_height), (self.width // 3, self.row_height * 2))
        paused = pygame_gui.elements.UIButton(
            relative_rect=rect,
            text="Play/Pause",
            manager=self.manager,
            container=self.panel,
        )
        paused.text_horiz_alignment = "left"
        self.inputs[paused] = "paused"

    def set_config_for_element(self, element, value=None):
        for elem, config_name in self.inputs.items():
            if elem == element:
                if value is None:
                    setattr(self.config, config_name, not getattr(self.config, config_name))
                else:
                    setattr(self.config, config_name, elem.i_type(value))


class IndividualMenu:

    def __init__(self, label, config: IndividualConfig, manager, menu_pos):
        self.config = config
        self.manager = manager
        self.panel = pygame_gui.elements.UIPanel(
            starting_layer_height=1,
            relative_rect=menu_pos,
            manager=manager)
        self.width = self.panel.rect.width
        self.half_width = self.width // 2 - 2
        self.row_height = 25
        self.label = label
        self.inputs = {}

    def _add_tooltip(self, rect, anchor, config_name):
        tooltip_button = pygame_gui.elements.UIButton(
            relative_rect=rect,
            text="?",
            manager=self.manager,
            container=self.panel,
            anchors={
                'left': 'left',
                'right': 'right',
                'top': 'bottom',
                'bottom': 'bottom',
                'bottom_target': anchor,
                'right_target': anchor,
            },
            tool_tip_text=f"text.individuals_tooltips_{config_name}",
            object_id=ObjectID(class_id="@tooltip_button", object_id=f"{self.__class__}{config_name}b"),
            starting_height=10,
        )

    def _input(self, label, rect:pygame.Rect, config_name, disable=False):
        label = pygame_gui.elements.UILabel(
            relative_rect=rect,
            text=label,
            manager=self.manager,
            container=self.panel,
            object_id=ObjectID(class_id='@left_alignment', object_id=f"{self.__class__}{config_name}l")
        )
        self._add_tooltip(pygame.Rect((100, 0), (25, 25)), label, config_name)
        entry = pygame_gui.elements.UITextEntryLine(
            relative_rect=rect.move(label.rect.width, 0),
            manager=self.manager,
            container=self.panel,
        )
        entry.i_type = int
        entry.set_text(str(getattr(self.config, config_name)))
        if disable:
            entry.disable()
        self.inputs[entry] = config_name
        return self.row_height

    def _input2(self, label, rect:pygame.Rect, config_name, disable=False):
        label = pygame_gui.elements.UILabel(
            relative_rect=rect,
            text=label,
            manager=self.manager,
            container=self.panel,
            object_id=ObjectID(class_id='@left_alignment', object_id=f"{self.__class__}{config_name}l")
        )
        self._add_tooltip(pygame.Rect((200, 0), (25, 25)), label, config_name)
        entry = pygame_gui.elements.UITextEntryLine(
            relative_rect=rect.move(0, self.row_height),
            manager=self.manager,
            container=self.panel,
        )
        entry.i_type = int
        entry.set_text(str(getattr(self.config, config_name)))
        if disable:
            entry.disable()

        self.inputs[entry] = config_name
        return self.row_height * 2

    def _slider(self, label, rect:pygame.Rect, config_name):
        label = pygame_gui.elements.UILabel(
            relative_rect=rect,
            text=label,
            manager=self.manager,
            container=self.panel,
            object_id=ObjectID(class_id='@left_alignment', object_id=f"{self.__class__}{config_name}l")
        )
        self._add_tooltip(pygame.Rect((200, 0), (25, 25)), label, config_name)
        entry = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=rect.move(0, self.row_height),
            manager=self.manager,
            container=self.panel,
            value_range=(0, 1),
            start_value=getattr(self.config, config_name)
        )
        entry.i_type = float
        self.inputs[entry] = config_name
        return self.row_height * 2

    def _multi_input(self, label, start_pos, config_name1, config_name2):
        label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(start_pos, (self.width, self.row_height)),
            text=label,
            manager=self.manager,
            container=self.panel,
            object_id=ObjectID(class_id='@left_alignment', object_id=f"{self.__class__}{config_name1}{config_name2}l")
        )
        rect = pygame.Rect(start_pos, (self.half_width, self.row_height))
        entry1 = pygame_gui.elements.UITextEntryLine(
            relative_rect=rect.move(0, self.row_height),
            manager=self.manager,
            container=self.panel,
        )
        entry1.i_type = int
        entry1.set_text(str(getattr(self.config, config_name1)))
        self._add_tooltip(pygame.Rect((100, 0), (25, 25)), entry1, config_name1)
        self.inputs[entry1] = config_name1
        entry2 = pygame_gui.elements.UITextEntryLine(
            relative_rect=rect.move(self.half_width, self.row_height),
            manager=self.manager,
            container=self.panel,
        )
        entry2.i_type = int
        entry2.set_text(str(getattr(self.config, config_name2)))
        self.inputs[entry2] = config_name2
        self._add_tooltip(pygame.Rect((225, 0), (25, 25)), entry1, config_name1)
        return self.row_height * 2

    def draw(self):
        pos = 0
        pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((0, pos), (self.width, self.row_height)),
            text=self.label,
            manager=self.manager,
            container=self.panel,
        )
        pos += self.row_height
        pos += self._input("Width", pygame.Rect((0, pos), (self.half_width, self.row_height)), "width", disable=True)
        pos += self._input("Height", pygame.Rect((0, pos), (self.half_width, self.row_height)), "height", disable=True)
        pos += self._input2("Population limit", pygame.Rect((0, pos), (self.width, self.row_height)), "population_limit", disable=True)
        pos += self._input("Rotation cost", pygame.Rect((0, pos), (self.half_width, self.row_height)), "rotate_drain")
        pos += self._input("Move cost", pygame.Rect((0, pos), (self.half_width, self.row_height)), "move_drain")
        pos += self._input("Max energy", pygame.Rect((0, pos), (self.half_width, self.row_height)), "max_energy")
        pos += self._input("Dead", pygame.Rect((0, pos), (self.half_width, self.row_height)), "dead")
        pos += self._input2("Crowded threshold", pygame.Rect((0, pos), (self.width, self.row_height)), "crowded_threshold")
        pos += self._input2("Hayflick limit", pygame.Rect((0, pos), (self.width, self.row_height)), "hayflick_limit")
        pos += self._slider("Mating%", pygame.Rect((0, pos), (self.width, self.row_height)), "mating_percent")
        pos += self._slider("Mutation%", pygame.Rect((0, pos), (self.width, self.row_height)), "mutation_chance")
        pos += self._slider("Reset%", pygame.Rect((0, pos), (self.width, self.row_height)), "reset_percent")
        pos += self._multi_input("Mating rect w x h", (0, pos),
                           "mating_rect_width", "mating_rect_height")
        pos += self._draw(pos)

        color_picker = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((0, pos), (self.width, self.row_height)),
            text="Color",
            manager=self.manager,
            container=self.panel,
            object_id=ObjectID(object_id=f"{id(self)}-color-button"))
        self.inputs[color_picker] = "color"

        return pos

    def set_config_for_element(self, element, value=None):
        for elem in list(self.inputs.keys()):
            if elem == element:
                config_name = self.inputs[elem]
                if config_name == 'color':
                    picker = pygame_gui.windows.UIColourPickerDialog(
                        pygame.Rect((200, 10), (300, 200)),
                        manager=self.manager,
                        window_title="Entity color",
                        initial_colour=pygame.Color(*self.config.color)
                    )
                    self.inputs[picker] = 'color_picked'
                elif config_name == 'color_picked':
                    element.kill()
                    self.inputs.pop(element)
                    self.config.color = value
                else:
                    setattr(self.config, config_name, elem.i_type(value))


class Individual():
    def __init__(self, world: WorldResources, config: IndividualConfig, tree=None, center=None):
        self.world = world
        self.config = config
        self.fissioned = False
        self.mated = False
        self.age = 0
        self.rect = pygame.Rect((0, 0, config.width, config.height))
        if center:
            self.rect.center = center
        else:
            self.rect.center = (random.randint(self.config.width, world.width - self.config.width),
                                random.randint(self.config.height, world.height - self.config.height))
        self.mating_rect = self.rect.inflate(self.config.mating_rect_width, self.config.mating_rect_height)

        self.energy = self.config.max_energy
        self.dir = random.randint(0, 3)

        self.pset = self.set_brain(self)
        if tree is None:
            self.tree = PrimitiveTree(self.expr_init())
        else:
            self.tree = copy.deepcopy(tree)

    def _dec_energy(self, val):
        self.energy -= val
        if self.energy < 0:
            self.energy = 0

    def _inc_energy(self, val):
        self.energy += val
        if self.energy > self.config.max_energy:
            self.energy = self.config.max_energy

    def copy(self):
        ret = self.__class__(world=self.world, config=self.config, tree=self.tree, center=self.rect.center)
        ret.energy = self.energy
        return ret

    @classmethod
    def mate(cls, ind1: 'Individual', ind2: 'Individual'):
        ind1.mated, ind2.mated = True, True
        ind1.age += 1
        ind2.age += 1
        tree1, tree2 = limited_crossover(copy.deepcopy(ind1.tree), copy.deepcopy(ind2.tree))
        child1 = ind1.copy()
        child1.tree = tree1
        child1.mated = True

        child2 = ind2.copy()
        child2.tree = tree1
        child2.mated = True
        return child1, child2

    def expr_init(self, pset=None, type_=None):
        if pset is None:
            return gp.genFull(self.pset, min_=1, max_=2, type_=type_)
        return gp.genFull(pset, min_=1, max_=2, type_=type_)

    def mutate(self):
        tree = limited_mutation(self.tree, expr=self.expr_init, pset=self.pset)
        self.age += 1
        self.tree = tree[0]

    def can_mate(self):
        """
        :return:
        """
        return not self.mated and self.age < self.config.hayflick_limit

    def rotate_left(self):
        self._dec_energy(self.config.rotate_drain)
        self.dir = (self.dir - 1) % 4

    def rotate_right(self):
        self._dec_energy(self.config.rotate_drain)
        self.dir = (self.dir + 1) % 4

    def _get_move_vector(self):
        current_pos = self.rect.center
        new_vector = [0, 0]
        match self.dir:
            case 0 if current_pos[1] > self.config.height:
                new_vector[1] -= self.config.height
            case 1 if current_pos[0] < self.world.width - self.config.width:
                new_vector[0] += self.config.width
            case 2 if current_pos[1] < self.world.height - self.config.height:
                new_vector[1] += self.config.height
            case 3 if current_pos[0] > self.config.width:
                new_vector[0] -= self.config.width
        return new_vector

    def move(self):
        self._dec_energy(self.config.move_drain)
        move = self._get_move_vector()
        self.rect.move_ip(*move)
        self.mating_rect.move_ip(*move)
        return move

    def run(self, routine):
        routine()

    def eval(self):
        return self.energy

    def get_color(self):
        dim = int(self.config.max_energy - self.energy)
        # print(f"Color: {self.config.color} for {id(self.config)} {self.config}")
        color = np.subtract(self.config.color, (dim, dim, dim, 0))
        color *= (color > 0)
        return list(color)


def eval_individual(ind):
    # Transform the tree expression to functional Python code
    routine = gp.compile(ind.tree, ind.pset)
    # Run the generated routine

    ind.run(routine)
    return ind


def vicinity_collision(left, right):
    if left != right:
        return left.mating_rect.colliderect(right.rect)
    else:
        return False


class Population(list):

    def __init__(self, config):
        super().__init__()
        self.config = config

    def eval(self):
        """
        Evaluate and update all entities
        :return:
        """
        pool = ThreadPool()
        new_population = pool.map(eval_individual, self, chunksize=500)
        self[:] = new_population

    def grow_population(self):
        """
                Mate the population aka create new entities if:
                There are entities which are in close proximity
                Algae.can_mate returned true
                :return:
                """
        children = []
        if 1 < len(self) < self.config.population_limit :
            # TODO this is time consuming with a lot of sprites, maybe we should pick the top x% and try to mate those?
            # maybe forget about proximity?
            can_mate = list(filter(lambda a: a.can_mate(), self))
            possible_mates = pygame.sprite.groupcollide(
                can_mate, can_mate, False, False,
                collided=vicinity_collision)

            mating_percent = self.config.mating_percent
            possible_mates_sorted = sorted(possible_mates.keys(), key=lambda x: x.eval(), reverse=True)
            possible_mates_sorted = possible_mates_sorted[:int(len(possible_mates_sorted) * mating_percent)]

            for base in possible_mates_sorted:
                mates = sorted([p for p in possible_mates[base] if p.can_mate()], key=lambda x: x.eval(), reverse=True)
                # this is like an if, mates can be emtpy
                for partner in mates:
                    if base.age > 10 or partner.age > 10:
                        print("ARENT WE OLD???")
                    offsprings = base.mate(base, partner)
                    children.extend(offsprings)
                    self.extend(offsprings)
                    break

        return children

    def reduce_population(self):
        """
        Reduce population with the rule: entity evals to entity.DEAD considered dead
        :return:
        """

        dead_entities = [a for a in self if a.eval() <= a.config.dead]
        for entity in dead_entities:
            self.remove(entity)

        return dead_entities

    def reset_mated(self):
        """
        Reset k% of the mated entities to ready to be mated again
        :param k:
        :return:
        """
        # reset mated for .1 of the mated
        mated = [a for a in self if a.mated]
        for ind in selection.selRandom(mated, k=int(len(mated) * self.config.reset_percent)):
            ind.mated = False

    def mutate_population(self):
        for ind in self:
            if random.random() <= ind.config.mutation_chance:
                ind.mutate()

    def as_group(self):
        ret = pygame.sprite.Group()
        for ind in self:
            s = pygame.sprite.Sprite()
            s.image = pygame.Surface((ind.config.width, ind.config.height))
            s.image.fill(ind.get_color())
            s.rect = ind.rect
            ret.add(s)
        return ret
