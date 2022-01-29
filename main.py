import contextlib
import dataclasses
import sys
import time

import PIL.ImageColor
import numpy as np
import pygame
import pygame_gui
from PIL import Image, ImageOps
from perlin_numpy import generate_perlin_noise_2d, generate_fractal_noise_2d

import creatures
from creatures.algae import AlgaeConfig, AlgaeMenu
from creatures.bacteria import BacteriaConfig, BacteriaMenu
from creatures.base import IndividualConfig, WorldResources, IndividualMenu, BottomPanel, GameConfig

APP_SIZE = 1024, 768
WORLD_SIZE = 512, 512
ALGAE_POPULATION = 200
BACTERIA_POPULATION = 100


def RGB_COLOR(color):
    return PIL.ImageColor.getrgb(PIL.ImageColor.colormap[color])


def get_noise(size, res):
    # np.random.seed(0)
    # return generate_perlin_noise_2d(SCREEN, res)
    return generate_fractal_noise_2d(size, res, 5)


def noise_to_normalized_np(noise):
    return ((noise + 1.0) / 2.0 * 255.0).astype(np.uint8)


def noise_to_image(noise, color):
    img = Image.fromarray(noise, mode='L')
    img = ImageOps.colorize(img, black='black', white=color)
    return img


def img_to_surface(img):
    return pygame.image.frombuffer(
        img.tobytes(), img.size, img.mode).convert()


@dataclasses.dataclass
class Display:
    window: pygame.Surface
    clock: pygame.time.Clock
    background: pygame.Surface
    algae_menu: IndividualMenu
    bacteria_menu: IndividualMenu
    manager: pygame_gui.UIManager
    bottom_panel: BottomPanel


def init(game_config, algae_config, bacteria_config):
    pygame.init()
    pygame.display.set_caption("Equilibrium")

    window = pygame.display.set_mode(APP_SIZE)
    clock = pygame.time.Clock()

    background = pygame.Surface(APP_SIZE)
    background.fill(pygame.Color('#330033'))

    manager = pygame_gui.UIManager(APP_SIZE)
    menu_width = (APP_SIZE[0] - WORLD_SIZE[0]) // 2

    algae_menu = AlgaeMenu("ALGAES", algae_config, manager, pygame.Rect((0, 0), (menu_width, APP_SIZE[1])))
    algae_menu.draw()
    bacteria_menu = BacteriaMenu("BACTERIAS", bacteria_config, manager, pygame.Rect((menu_width + WORLD_SIZE[0], 0), (menu_width, APP_SIZE[1])))
    bacteria_menu.draw()

    bottom_panel = BottomPanel(label="", config=game_config, manager=manager, menu_pos=pygame.Rect(menu_width, WORLD_SIZE[1], WORLD_SIZE[0], APP_SIZE[1] - WORLD_SIZE[1]))
    bottom_panel.draw()

    return Display(window=window, clock=clock, background=background, manager=manager,
                   algae_menu=algae_menu, bacteria_menu=bacteria_menu, bottom_panel=bottom_panel)


def display_text(text, screen, pos):
    font = pygame.font.Font(pygame.font.get_default_font(), 16)
    text = font.render(text, True, (255, 255, 255))
    screen.blit(text, dest=pos)


@contextlib.contextmanager
def timer(name):
    tic = time.perf_counter()
    yield
    toc = time.perf_counter()
    diff = toc - tic
    if diff > 0.5:
        print(f"{name} took {toc - tic:0.4f}s")


def main():
    algae_config = AlgaeConfig(
        width=2,
        height=2,
        max_energy=100,
        rotate_drain=1,
        move_drain=2,
        photosynthesis_gain=100,
        mating_percent=.1,
        mutation_chance=.1,
        crowded_threshold=3,
        population_limit=1000,
        reset_percent=.1,
        dead=0,
        mating_rect_width=6,
        mating_rect_height=6,
        hayflick_limit=10)

    bacteria_config = BacteriaConfig(
        width=2,
        height=2,
        max_energy=100,
        rotate_drain=0,
        move_drain=2,
        mating_percent=.2,
        mutation_chance=.1,
        crowded_threshold=5,
        population_limit=1000,
        reset_percent=.1,
        dead=0,
        mating_rect_width=12,
        mating_rect_height=12,
        food_sensing_distance=12,
        hayflick_limit=10)

    game_config = GameConfig(
        game_speed=25,
        algae_population=ALGAE_POPULATION,
        bacteria_population=BACTERIA_POPULATION,
        paused=False,
        generation=0
    )

    display = init(game_config, algae_config, bacteria_config)
    water = noise_to_normalized_np(get_noise(WORLD_SIZE, (2, 2)))
    light = noise_to_normalized_np(get_noise(WORLD_SIZE, (4, 4)))

    world_config = WorldResources(height=WORLD_SIZE[1], width=WORLD_SIZE[0], water=water, light=light)

    water_img = noise_to_image(water, 'blue')
    light_img = noise_to_image(light, 'red')

    world_img = Image.blend(water_img, light_img, alpha=.5)
    world_surface = img_to_surface(world_img)

    all_individuals = []
    all_algae = creatures.AlgaeGroup(config=algae_config)
    for _ in range(game_config.algae_population):
        a = creatures.Algae(world_config, algae_config)
        all_algae.append(a)
    all_individuals.extend(all_algae)

    all_bacteria = creatures.BacteriaGroup(config=bacteria_config)
    for _ in range(game_config.bacteria_population):
        b = creatures.Bacteria(world_config, bacteria_config, individuals=all_individuals)
        all_bacteria.append(b)
    all_individuals.extend(all_bacteria)

    while True:
        for event in pygame.event.get():
            match event.type:
                case pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                case pygame_gui.UI_TEXT_ENTRY_FINISHED:
                    display.algae_menu.set_config_for_element(event.ui_element, event.text)
                    display.bacteria_menu.set_config_for_element(event.ui_element, event.text)
                case pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
                    print(event.value)
                    display.algae_menu.set_config_for_element(event.ui_element, event.value)
                    display.bacteria_menu.set_config_for_element(event.ui_element, event.value)
                    display.bottom_panel.set_config_for_element(event.ui_element, event.value)
                case pygame_gui.UI_BUTTON_PRESSED:
                    display.bottom_panel.set_config_for_element(event.ui_element)
                case pygame.MOUSEBUTTONUP:
                    pos = pygame.sprite.DirtySprite()
                    x, y = event.pos[0] - 2, event.pos[1] - 2
                    pos.rect = pygame.Rect(x - (APP_SIZE[0] - WORLD_SIZE[0]) // 2, y, 4, 4)
                    print(pos.rect)
                    match = pygame.sprite.spritecollideany(pos, all_individuals)
                    if match:
                        world = world_surface.get_at(match.rect.center)
                        print(f"Sun: {world[0] / 255} Water: {world[2] / 255}")
                        print(f"Energy: {match.eval()}")
                        print(f"Cell age: {match.age}")
                        print(f"Dead?: {match.eval() == match.config.dead}")
                        # match.save_tree()
                    else:
                        print("no match")

            display.manager.process_events(event)
        display.clock.tick(game_config.game_speed)
        display.manager.update(25 / 1000)
        if not game_config.paused:
            game_config.generation += 1
            with timer("eval "):
                # eval
                all_algae.eval()
                all_bacteria.eval()

            with timer("carn has eaten "):
                food = pygame.sprite.groupcollide(all_bacteria, all_algae, False, False)
                for _, algaes in food.items():
                    for algae in algaes:
                        if algae in all_algae:
                            all_algae.remove(algae)
                            all_individuals.remove(algae)

            with timer("die "):
                # die
                dead_algae = all_algae.reduce_population()
                map(lambda d: all_individuals.remove(d), dead_algae)
                all_bacteria.reduce_population()
                map(lambda d: all_individuals.remove(d), dead_algae)

            with timer("mate "):
                # mate
                new_algae = all_algae.grow_population()
                all_individuals.extend(new_algae)
                new_bacteria = all_bacteria.grow_population()
                all_individuals.extend(new_bacteria)

            with timer("mutate "):
                # mutate
                all_algae.mutate_population()
                all_bacteria.mutate_population()

            with timer("reset "):
                all_algae.reset_mated()
                all_bacteria.reset_mated()

            with timer("blit + type "):
                sprite_surface = pygame.Surface(WORLD_SIZE, pygame.SRCALPHA)
                display.window.blit(world_surface, world_surface.get_rect(topleft=((APP_SIZE[0] - WORLD_SIZE[0]) // 2, 0)))
                display.bottom_panel.algae_count.set_text(str(len(all_algae)))
                display.bottom_panel.bacteria_count.set_text(str(len(all_bacteria)))
                display.bottom_panel.generation_count.set_text(str(game_config.generation))
                all_algae.as_group().draw(sprite_surface)
                all_bacteria.as_group().draw(sprite_surface)

        display.window.blit(sprite_surface, sprite_surface.get_rect(topleft=((APP_SIZE[0] - WORLD_SIZE[0]) // 2, 0)))
        display.manager.draw_ui(display.window)
        pygame.display.flip()


if __name__ == '__main__':
    main()
