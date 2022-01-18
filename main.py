import copy
import random
import sys

import PIL.ImageColor
import numpy as np
import pygame
from PIL import Image, ImageOps
from deap import base, tools, gp
from perlin_numpy import generate_perlin_noise_2d, generate_fractal_noise_2d

from creatures.algae import Algae, eval_algae

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

    toolbox =  base.Toolbox()
    # toolbox.register('algae', tools.initRepeat, Algae, n=1)
    toolbox.register('evaluate', Algae.eval)

    return window, clock, toolbox


def display_text(text, screen, pos):
    font = pygame.font.Font(pygame.font.get_default_font(), 16)
    text = font.render(text, True, (255, 255, 255))
    screen.blit(text, dest=pos)


def vicinity_collision(left, right):
    if left != right:
        return left.mating_rect.colliderect(right.rect)
    else:
        return False


def main():
    window, clock, toolbox = init()
    water = get_noise((2, 2))
    water_img = noise_to_image(water, 'blue')

    light = get_noise((4, 4))
    light_img = noise_to_image(light, 'red')

    world_img = Image.blend(water_img, light_img, alpha=.5)
    world_surface = img_to_surface(world_img)

    all_sprites = pygame.sprite.Group()
    all_algae = pygame.sprite.Group()
    for _ in range(ALGAE_POPULATION):
        algae = Algae(world_surface)
        all_sprites.add(algae)
        all_algae.add(algae)

    gen = 0
    while True:
        clock.tick(1)
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


        # window.fill(0)
        window.blit(world_surface, world_surface.get_rect(topleft=(0, 0)))
        display_text(f"Algae count: {len(all_algae)}", window, pos=(0,0))
        display_text(f"Generation count: {gen}", window, pos=(0, SCREEN[1] - 16))

        for entity in all_algae:
            res = eval_algae(entity.tree, entity.pset, entity)
            tree = entity.mutate()
            temp = entity.copy()
            res2 = eval_algae(tree, entity.pset, temp)
            # print(f"{res=} {res2=}")
            if res2 > res:
                entity.pset = temp.pset
                entity.tree = temp.tree
            temp.kill()

            entity.fitness.values = toolbox.evaluate(entity)
            entity.update_color()

        # die
        dead_algae = [a for a in all_sprites if a.eval() == Algae.DEAD]
        for algae in dead_algae:
            algae.kill()

        # mate
        possible_mates = pygame.sprite.groupcollide(
            all_algae, all_algae, False, False,
            collided=vicinity_collision)
        for base, mates in possible_mates.items():
            if base.eval() > base.MATING_LIMIT and not base.mated:
                possible_partners = [p for p in mates if p.eval() > p.MATING_LIMIT and not p.mated and len(base.tree) == len(p.tree)]
                if possible_partners:
                    pair = base, random.choice(possible_partners)
                    pair[0].mated = True
                    pair[1].mated = True
                    print(f"{len(base.tree)} {len(pair[1].tree)}")
                    children = tools.cxOnePoint(toolbox.clone(pair[0].tree), toolbox.clone(pair[1].tree))
                    choice = random.randint(0, 1)
                    child = Algae(world_surface, pset=pair[choice].pset, tree=children[choice])
                    all_algae.add(child)
                    all_sprites.add(child)

        all_sprites.draw(window)
        # print(len(all_sprites))

        gen += 1
        pygame.display.flip()


if __name__ == '__main__':
    main()
