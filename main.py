import contextlib
import multiprocessing
import random
import sys
import time

import PIL.ImageColor
import numpy as np
import pygame
from PIL import Image, ImageOps
from deap.base import Toolbox
from perlin_numpy import generate_perlin_noise_2d, generate_fractal_noise_2d

import creatures
from creatures import Algae, Bacteria
from creatures.base import IndividualConfig

SCREEN = 512, 512
ALGAE_POPULATION = 200
BACTERIA_POPULATION = 100


def RGB_COLOR(color):
    return PIL.ImageColor.getrgb(PIL.ImageColor.colormap[color])


def get_noise(res):
    # np.random.seed(0)
    # return generate_perlin_noise_2d(SCREEN, res)
    return generate_fractal_noise_2d(SCREEN, res, 5)


def noise_to_image(noise, color):
    # print(dir(cm))
    img = Image.fromarray(((noise + 1.0) / 2.0 * 255.0).astype(np.uint8), mode='L')
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
    water = get_noise((2, 2))
    water_img = noise_to_image(water, 'blue')

    light = get_noise((4, 4))
    light_img = noise_to_image(light, 'red')

    world_img = Image.blend(water_img, light_img, alpha=.5)
    world_surface = img_to_surface(world_img)

    algae_config = IndividualConfig(
        width=2,
        height=2,
        rotate_drain=1,
        move_drain=2,
        photosynthesis_gain=100,
        mating_percent=.1,
        mutation_chance=.1,
        crowded_threshold=3,
        reset_percent=.01,
        dead=0,
        mating_rect_width=6,
        mating_rect_height=6)

    bacteria_config = IndividualConfig(
        width=2,
        height=2,
        rotate_drain=1,
        move_drain=2,
        photosynthesis_gain=100,
        mating_percent=.5,
        mutation_chance=.1,
        crowded_threshold=5,
        reset_percent=.01,
        dead=0,
        mating_rect_width=12,
        mating_rect_height=12)

    all_sprites = pygame.sprite.Group()
    all_algae = creatures.AlgaeGroup(config=algae_config)
    for _ in range(ALGAE_POPULATION):
        a = creatures.Algae(world_surface, algae_config)
        all_sprites.add(a)
        all_algae.add(a)

    all_bacteria = creatures.BacteriaGroup(config=bacteria_config)
    for _ in range(BACTERIA_POPULATION):
        b = creatures.Bacteria(world_surface, bacteria_config, sprites=all_sprites)
        all_bacteria.add(b)
        all_sprites.add(b)

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
                    match = pygame.sprite.spritecollideany(pos, all_sprites)
                    if match:
                        world = world_surface.get_at(match.rect.center)
                        print(f"{match.energy}")
                        print(f"Sun: {world[0] / 255} Water: {world[2] / 255}")
                        print(f"Color: {match.image.get_at((1,1))}")
                        match.save_tree()

        with timer("eval "):
            # eval
            all_algae.eval()
            all_bacteria.eval()

        with timer("die "):
            # die
            all_algae.reduce_population()
            all_bacteria.reduce_population()

        with timer("mate "):
            # mate
            new_algae = all_algae.grow_population()
            all_sprites.add(new_algae)
            new_bacteria = all_bacteria.grow_population()
            all_sprites.add(new_bacteria)

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
            all_sprites.draw(window)
            pygame.display.flip()


if __name__ == '__main__':
    main()
