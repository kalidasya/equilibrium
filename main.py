import sys

import PIL.ImageColor
import numpy as np
import pygame
from PIL import Image, ImageOps
from deap import creator, base, tools, gp
from perlin_numpy import generate_perlin_noise_2d, generate_fractal_noise_2d

from creatures.algae import Algae, eval_algae

SCREEN = 512, 512
ALGAE_POPULATION = 1


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
    # toolbox.register('evaluate', Algae.eval)

    return window, clock, toolbox


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
                        print(f"{match.fitness} {match.dryness=} {match.paleness=}")
                        print(f"Sun: {world[0] / 255} Water: {world[2] / 255}")
                        print(f"Color: {match.image.get_at((1,1))}")
                        match.save_tree()


        # window.fill(0)
        window.blit(world_surface, world_surface.get_rect(topleft=(0, 0)))

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

            entity.update_color()
            # match random.randint(0, 4):
            #     case 0:
            #         entity.photosynthesise()
            #     case 1:
            #         entity.rotate_left()
            #     case 2:
            #         entity.rotate_right()
            #     case 3:
            #         if entity.lighter_ahead():
            #             entity.move()

            # fitness = toolbox.evaluate(entity)
            # if fitness != (1,):
            #     print(fitness)
            # entity.fitness.values = fitness

        # fitnesses = list(map(toolbox.evaluate, all_sprites))
        # print(fitnesses)
        # for ind, fit in zip(all_sprites, fitnesses):
        #     ind.fitness = fit

        # for s in all_sprites:
        #     print(s.fitness)
        #     if not s.fitness:
        #         s.kill()

        all_sprites.draw(window)
        # print(len(all_sprites))

        pygame.display.flip()


if __name__ == '__main__':
    main()
