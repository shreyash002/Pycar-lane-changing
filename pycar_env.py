import pygame as pg
from random import randint
import PIL.Image as Image
import numpy as np


class PyCar():
    def __init__(self):
        super().__init__()
        pg.font.init()
        pg.mixer.init()


    def player(self):
        self.screen.blit(self.cars_img[0], (self.player_x - int(self.car_x / 2), self.player_y - int(self.car_y / 2)))


    def enemies(self):
        #screen.blit(cars_img[1], (enemies_x - 115, enemy1_y))
        #screen.blit(cars_img[2], (enemies_x - int(car_x / 2), enemy2_y))
        #screen.blit(cars_img[3], (enemies_x + 60, enemy3_y))
        self.screen.blit(self.cars_img[1], (self.enemy1_x - int(self.car_x / 2), self.enemy1_y - int(self.car_y / 2)))
        self.screen.blit(self.cars_img[2], (self.enemy2_x - int(self.car_x / 2), self.enemy2_y - int(self.car_y / 2)))
        self.screen.blit(self.cars_img[3], (self.enemy3_x - int(self.car_x / 2), self.enemy3_y - int(self.car_y / 2)))


    def text(self, txt_msg, txt_color, txt_size, txt_x, txt_y):
        font = pg.font.SysFont('arial', txt_size, True)
        txt = font.render(txt_msg, True, txt_color)
        self.screen.blit(txt, (txt_x, txt_y))

    def reset_game(self):
        'COLORS (RGB)'
        self.white = (255, 255, 255)
        self.black = (0, 0, 0)
        self.red = (255, 0, 0)

        'WINDOW'
        self.wid, self.hei = 400, 500
        self.screen = pg.display.set_mode((self.wid, self.hei))
        pg.display.set_caption('PyCar')

        'SPRITE SIZE'
        self.car_x, self.car_y = 54, 94

        'PLAYER'
        self.player_x, self.player_y = int(self.wid / 2), int(self.hei - 50)
        self.player_spd = 5

        'ENEMIES'
        self.enemies_x = int(self.wid / 2) 
        self.enemy1_x = int(self.wid / 2) - 80
        self.enemy2_x = int(self.wid / 2) 
        self.enemy3_x = int(self.wid / 2) + 80

        self.enemy1_y = randint(-self.hei, -self.car_y)  # Posição do carro vermelho (lado esquerdo).
        self.enemy2_y = randint(-self.hei, -self.car_y)  # Posição do carro amarelo (lado direito).
        self.enemy3_y = randint(-self.hei, -self.car_y)  # Posição do carro azul (centro).
        self.enemies_spd = 0

        'IMAGES'
        self.bg = pg.image.load('assets/images/background/Road.png').convert()
        self.bg = pg.transform.scale(self.bg, (self.wid, self.hei))
        self.bg_y = 0
        self.cars_img = [pg.image.load('assets/images/sprites/Player_Car.png'),
                    pg.image.load('assets/images/sprites/Enemy1_Car.png'),
                    pg.image.load('assets/images/sprites/Enemy2_Car.png'),
                    pg.image.load('assets/images/sprites/Enemy3_Car.png')]

        'MUSIC'
        pg.mixer_music.load('assets/sounds/music/Chillwave_Nightdrive.mp3')
        pg.mixer_music.play(-1)

        'SOUND EFFECT'
        self.car_collision = pg.mixer.Sound('assets/sounds/sound_effects/Car_Collision.wav')

        self.clock = pg.time.Clock()
        self.score = 1
        self.score_spd = 0

    def run_game(self, keyboard=True):
        main = True
        while main:
            self.clock.tick(60)
            # bg_y = 0
            # --- Faz com que a imagem de background se repita, deslizando de cima para baixo --- #
            bg_y1 = self.bg_y % self.bg.get_height()
            self.bg_y += 3
            self.screen.blit(self.bg, (0, bg_y1 - self.bg.get_height()))
            if bg_y1 < self.hei:
                self.screen.blit(self.bg, (0, bg_y1))

            self.player()
            self.enemies()

            pg.draw.rect(self.screen, self.white, (0, 0, 54, 20))
            self.text('S: ' + str(self.score), self.black, 15, 0, 0)

            for event in pg.event.get():
                if event.type == pg.QUIT:
                    main = False

            'CONTROLS'
            arrows = pg.key.get_pressed()
            if arrows[pg.K_RIGHT] and self.player_x <= 290:
                self.player_x += self.player_spd
            if arrows[pg.K_LEFT] and self.player_x >= 110:
                self.player_x -= self.player_spd
            if arrows[pg.K_UP] and self.player_y >= int(self.car_y/2) + 50:
                self.player_y -= self.player_spd
            if arrows[pg.K_DOWN] and self.player_y <= int(self.hei - 50):
                self.player_y += self.player_spd
            
            'ENEMIES SPEED'
            self.enemy1_y += self.enemies_spd + 5
            self.enemy2_y += self.enemies_spd + 2
            self.enemy3_y += self.enemies_spd + 4

            # --- Os inimigos aparecem, aleatoriamente, fora do background após sair do mesmo --- #
            if self.enemy1_y > self.hei:
                self.enemy1_y = randint(-2500, - 2000)
            if self.enemy2_y > self.hei:
                self.enemy2_y = randint(-1000, -750)
            if self.enemy3_y > self.hei:
                self.enemy3_y = randint(-1750, -1250)

            'SCORE'
            if self.score_spd <= 60:
                self.score_spd += 1
            else:
                self.score += 1
                self.score_spd = 0

            'COLLISION'
            if abs(self.player_x - self.enemy3_x ) < self.car_x  and abs(self.player_y - self.enemy3_y) < self.car_y + 5:  # Lado direito.
                print("Enemy 3 collision")
                self.car_collision.play()
                self.score -= 10
                self.enemy3_y = randint(-1750, -1250)
            if  abs(self.player_x - self.enemy1_x ) < self.car_x  and abs(self.player_y - self.enemy1_y) < self.car_y + 5 :  # Lado esquerdo.
                print("Enemy 1 collision")
                self.car_collision.play()
                self.score -= 10
                self.enemy1_y = randint(-2500, - 2000)
            if abs(self.player_x - self.enemy2_x ) < self.car_x  and abs(self.player_y - self.enemy2_y) < self.car_y + 5 :  # Centro.
                print("Enemy 2 collision")
                #if player_x + 40 > enemies_x - 10 and abs(player_y - enemy2_y) < 50:
                self.car_collision.play()
                self.score -= 10
                self.enemy2_y = randint(-1000, -750)

            if self.score <= 0:
                break

            pg.display.update()
            pg.image.save(self.screen, "screenshot.jpeg")
            img = np.array(Image.open("screenshot.jpeg"))
            #print(img.shape)

        pg.quit()


if __name__ == "__main__":
    pc = PyCar()
    pc.reset_game()
    pc.run_game()