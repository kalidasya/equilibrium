import contextlib
import multiprocessing
import sys
import time

import PIL.ImageColor
import numpy as np
import pygame
from PIL import Image, ImageOps
from deap.base import Toolbox
from perlin_numpy import generate_perlin_noise_2d, generate_fractal_noise_2d

import creatures
from creatures.base import IndividualConfig, WorldResources

SCREEN = 512, 512
ALGAE_POPULATION = 200
BACTERIA_POPULATION = 100


def RGB_COLOR(color):
    return PIL.ImageColor.getrgb(PIL.ImageColor.colormap[color])


def get_noise(res):
    # np.random.seed(0)
    # return generate_perlin_noise_2d(SCREEN, res)
    return generate_fractal_noise_2d(SCREEN, res, 5)


def noise_to_normalized_np(noise):
    return ((noise + 1.0) / 2.0 * 255.0).astype(np.uint8)


def noise_to_image(noise, color):
    img = Image.fromarray(noise, mode='L')
    img = ImageOps.colorize(img, black='black', white=color)
    return img


def img_to_surface(img):
    return pygame.image.frombuffer(
        img.tobytes(), img.size, img.mode).convert()


def init():
    pygame.init()
    window = pygame.display.set_mode(SCREEN)
    clock = pygame.time.Clock()
    toolbox = Toolbox()
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    return window, clock


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
    window, clock = init()
    water = noise_to_normalized_np(get_noise((2, 2)))
    light = noise_to_normalized_np(get_noise((4, 4)))

    world_config = WorldResources(height=SCREEN[1], width=SCREEN[0], water=water, light=light)

    water_img = noise_to_image(water, 'blue')
    light_img = noise_to_image(light, 'red')

    world_img = Image.blend(water_img, light_img, alpha=.5)
    world_surface = img_to_surface(world_img)

    algae_config = IndividualConfig(
        width=2,
        height=2,
        rotate_drain=1,
        move_drain=2,
        photosynthesis_gain=100,
        mating_percent=.9,
        mutation_chance=.1,
        crowded_threshold=3,
        population_limit=1000,
        reset_percent=.3,
        dead=0,
        mating_rect_width=6,
        mating_rect_height=6,
        hayflick_limit=50)

    bacteria_config = IndividualConfig(
        width=2,
        height=2,
        rotate_drain=0,
        move_drain=2,
        photosynthesis_gain=100,
        mating_percent=.5,
        mutation_chance=.9,
        crowded_threshold=5,
        population_limit=1000,
        reset_percent=.1,
        dead=0,
        mating_rect_width=12,
        mating_rect_height=12,
        food_sensing_distance=12,
        hayflick_limit=50)

    all_individuals = []
    all_algae = creatures.AlgaeGroup(config=algae_config)
    for _ in range(ALGAE_POPULATION):
        a = creatures.Algae(world_config, algae_config)
        all_algae.append(a)
    all_individuals.extend(all_algae)

    all_bacteria = creatures.BacteriaGroup(config=bacteria_config)
    for _ in range(BACTERIA_POPULATION):
        b = creatures.Bacteria(world_config, bacteria_config, individuals=all_individuals)
        all_bacteria.append(b)
    all_individuals.extend(all_bacteria)

    gen = 0
    while True:
        clock.tick(25)
        for event in pygame.event.get():
            match event.type:
                case pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                case pygame.MOUSEBUTTONUP:
                    pos = pygame.sprite.DirtySprite()
                    x, y = event.pos[0] - 2, event.pos[1] - 2
                    pos.rect = pygame.Rect(x, y, 4, 4)
                    match = pygame.sprite.spritecollideany(pos, all_individuals)
                    if match:
                        world = world_surface.get_at(match.rect.center)
                        print(f"Sun: {world[0] / 255} Water: {world[2] / 255}")
                        print(f"Value: {match.eval()}")
                        print(f"Dead?: {match.eval() == match.config.dead}")
                        match.save_tree()

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
            window.blit(world_surface, world_surface.get_rect(topleft=(0, 0)))
            display_text(f"Algae count: {len(all_algae)}", window, pos=(0, 0))
            display_text(f"Bacteria count: {len(all_bacteria)}", window, pos=(0, 20))
            display_text(f"Generation count: {gen}", window, pos=(0, SCREEN[1] - 16))

        gen += 1

        # draw
        with timer("drawing "):
            g = all_algae.as_group()
            g.draw(window)
            all_bacteria.as_group().draw(window)
            pygame.display.flip()


if __name__ == '__main__':
    main()
