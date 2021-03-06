import dataclasses
import html
from functools import partial

import numpy as np
import pygame
from deap import gp
from yapf.yapflib.yapf_api import FormatCode

from creatures import base
from creatures.base import IndividualConfig, IndividualMenu
from utils import func


@dataclasses.dataclass
class AlgaeConfig(IndividualConfig):
    photosynthesis_gain: int


class AlgaeMenu(IndividualMenu):

    def _draw(self, pos):
        return self._input2("Photosynthesis gain", pygame.Rect((0, pos), (self.width, self.row_height)),
                            "photosynthesis_gain")


class Algae(base.Individual):
    name = "Bacteria"

    @classmethod
    def set_brain(cls, algae):
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

    def _get_light_level(self, coords=None):
        if coords is None:
            coords = self.rect.center

        sun = self.world.light[coords[0], coords[1]]
        return sun / 255

    def _get_water_level(self, coords=None):
        if coords is None:
            coords = self.rect.center
        water = self.world.water[coords[0], coords[1]]
        return water / 255

    def photosynthesize(self):
        efficiency = min(self._get_light_level(), self._get_water_level())
        # print(f"Gaining {int(self.PHOTOSYNTHESIS_GAIN * efficiency)} energy with {efficiency=}")
        self._inc_energy(self.config.photosynthesis_gain * efficiency)

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

    def get_description(self):
        formatted_code, _ = FormatCode(str(self.tree))
        return f"""<b>{html.escape(str(self))} entity found</b><br/>
Sun level: {round(self._get_light_level(), 2)}
Water: {round(self._get_water_level(), 2)}
Energy: {self.eval()}
Cell age: {self.age}<
<hr/>
{formatted_code}""".replace("\n", "<br/>")

    def __str__(self):
        return f"Algae {id(self)} {self.rect}"


class AlgaeGroup(base.Population):
    pass
