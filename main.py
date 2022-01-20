import contextlib
import random
import sys
import time

import PIL.ImageColor
import numpy as np
import pygame
from PIL import Image, ImageOps
from deap import base, tools, gp
from perlin_numpy import generate_perlin_noise_2d, generate_fractal_noise_2d

import creatures

SCREEN = 512, 512
ALGAE_POPULATION = 200


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

    toolbox = base.Toolbox()
    # toolbox.register('algae', tools.initRepeat, Algae, n=1)
    # toolbox.register('evaluate', Algae.eval)

    return window, clock, toolbox


def display_text(text, screen, pos):
    font = pygame.font.Font(pygame.font.get_default_font(), 16)
    text = font.render(text, True, (255, 255, 255))
    screen.blit(text, dest=pos)


@contextlib.contextmanager
def timer(name):
    tic = time.perf_counter()
    yield
    toc = time.perf_counter()
    print(f"{name} took {toc - tic:0.4f}s")


def main():
    window, clock, toolbox = init()
    water = get_noise((2, 2))
    water_img = noise_to_image(water, 'blue')

    light = get_noise((4, 4))
    light_img = noise_to_image(light, 'red')

    world_img = Image.blend(water_img, light_img, alpha=.5)
    world_surface = img_to_surface(world_img)

    all_sprites = pygame.sprite.Group()
    all_algae = creatures.AlgaeGroup()
    for _ in range(ALGAE_POPULATION):
        a = creatures.Algae(world_surface)
        all_sprites.add(a)
        all_algae.add(a)

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

        # eval
        all_algae.eval_population()

        # die
        all_algae.reduce_population()

        # mate
        new_algae = all_algae.mate_population()
        all_sprites.add(new_algae)

        # mutate
        for algae in all_algae:
            if random.random() <= algae.MUTATION_CHANCE:
                algae.mutate()

        all_algae.reset_mated(10)

        window.blit(world_surface, world_surface.get_rect(topleft=(0, 0)))
        display_text(f"Algae count: {len(all_algae)}", window, pos=(0, 0))
        display_text(f"Generation count: {gen}", window, pos=(0, SCREEN[1] - 16))

        gen += 1

        # draw
        with timer("drawing "):
            all_sprites.draw(window)
            pygame.display.flip()


if __name__ == '__main__':
    main()
