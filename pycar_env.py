import pygame as pg
from random import randint
import PIL.Image as Image
import numpy as np
import time


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
        self.player_spd = 25
        self.player_spd_x = self.player_spd #85
        self.player_spd_y = self.player_spd #20

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
                    pg.image.load('assets/images/sprites/Enemy2_Car.png'),
                    pg.image.load('assets/images/sprites/Enemy2_Car.png'),
                    pg.image.load('assets/images/sprites/Enemy2_Car.png')]

        'MUSIC'
        # pg.mixer_music.load('assets/sounds/music/Chillwave_Nightdrive.mp3')
        # pg.mixer_music.play(-1)

        'SOUND EFFECT'
        # self.car_collision = pg.mixer.Sound('assets/sounds/sound_effects/Car_Collision.wav')

        self.clock = pg.time.Clock()
        self.score = 0
        self.score_spd = 0
        self.prev_lane = "center"

    def get_keyboard(self):
        return pg.key.get_pressed()

    def step(self, action):
        # actions:
        # 0 --> Nothing
        # 1 --> Up
        # 2 --> Down
        # 3 --> Left
        # 4 --> Right
        
        collision = False

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

        #pg.draw.rect(self.screen, self.white, (0, 0, 54, 20))
        #self.text('S: ' + str(self.score), self.black, 15, 0, 0)

        for event in pg.event.get():
            if event.type == pg.QUIT:
                main = False

        'CONTROLS'
        
        if action == 4: #and self.player_x <= 290:
            self.player_x += self.player_spd_x
        if action == 3: # and self.player_x >= 110:
            self.player_x -= self.player_spd_x
        if action == 2 and self.player_y >= int(self.car_y/2) + 50:
            self.player_y -= self.player_spd_y
        if action == 1 and self.player_y <= int(self.hei - 50):
            self.player_y += self.player_spd_y
        else:
            pass

        if self.player_x > 310:
            collision = True
        elif self.player_x < 90:
            collision = True

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

        # 'SCORE'
        # if self.score_spd <= 60:
        #     self.score_spd += 1
        # else:
        #     self.score += 1
        #     self.score_spd = 0

        'COLLISION'
        if abs(self.player_x - self.enemy3_x ) < self.car_x  and abs(self.player_y - self.enemy3_y) < self.car_y + 5:  # Lado direito.
            print("Enemy 3 collision")
            # self.car_collision.play()
            collision = True
            # self.score -= 10
            self.enemy3_y = randint(-1750, -1250)
        if  abs(self.player_x - self.enemy1_x ) < self.car_x  and abs(self.player_y - self.enemy1_y) < self.car_y + 5 :  # Lado esquerdo.
            print("Enemy 1 collision")
            # self.car_collision.play()
            collision = True
            # self.score -= 10
            self.enemy1_y = randint(-2500, - 2000)
        if abs(self.player_x - self.enemy2_x ) < self.car_x  and abs(self.player_y - self.enemy2_y) < self.car_y + 5 :  # Centro.
            print("Enemy 2 collision")
            #if player_x + 40 > enemies_x - 10 and abs(player_y - enemy2_y) < 50:
            # self.car_collision.play()
            collision = True
            # self.score -= 10
            self.enemy2_y = randint(-1000, -750)
        
        'LANE'
        ##action 3 is left and action 4 is right
        # action = 4 ##NEED TO FIX THIS
        
        lane_changed = False
        if  self.prev_lane == "center" and (action == 3 or action == 4): 
            if abs(self.player_x-int(self.wid / 2)) > 75 and abs(self.player_x-int(self.wid / 2)) < 120:
                lane_changed = True
        elif self.prev_lane == "left":
            #if action == 3:
            #    collision = True
            if action == 4 and abs(self.player_x-int(self.wid / 2)) < 15:
                lane_changed = True
            else:
                pass
        elif self.prev_lane == "right":
            #if action == 4:
            #    collision = True
            if action == 3 and abs(self.player_x-int(self.wid / 2)) < 15:
                lane_changed = True
            else:
                pass
        else:
            pass    
        
        if lane_changed:
            self.prev_lane = self.get_current_lane()
            # print(prev_lane) 

        'NEW SCORE'
        reward = self.get_reward(collision, lane_changed, self.get_current_lane()==None)
        self.score += reward
        # print(self.score)
        done = False
        if collision:
            done = True

        pg.display.update()
        # image_data = pg.surfarray.array3d(pg.display.get_surface())

        # pg.image.save(self.screen, "screenshot.jpeg")
        # img = np.array(Image.open("screenshot.jpeg"))
        # #print(img.shape)
        return None, reward, done

    def get_state(self):
        num_images = 4
        images=np.zeros((100, 125, 12))
        for i in range(num_images):
            pg.display.update()
            img = Image.fromarray(pg.surfarray.array3d(pg.display.get_surface())).resize(size=(125, 100))
            images[:, :, 3*i:3*i+3] = np.array(img)
        return images

    def run_game(self, keyboard=True):
        main = True
        while main:
            done = False
            if keyboard:
                action = 0
                arrows = self.get_keyboard()
                if arrows[pg.K_RIGHT]: #and self.player_x <= 290:
                    action = 4
                if arrows[pg.K_LEFT]: # and self.player_x >= 110:
                    action = 3
                if arrows[pg.K_UP]:
                    action = 1
                if arrows[pg.K_DOWN]:
                    action = 2            
                state, reward, done = self.step(action)
                time.sleep(0.05)
            if done:
                break

        pg.quit()

    def get_reward(self, collision, lane_changed, rule):

        r_col = -20 if collision else 0
        r_comp = 0 if lane_changed else 0

        # No rule for now
        # r_vel = -alpha*np.abs(v_target-v_agent)

        # r_safe = ##TODO

        if rule:
            r_const = -1
        else:
            r_const = 1

        # r_rule = r_vel + r_safe + r_const

        return r_col+r_comp + r_const
    
    def get_current_lane(self):
        if abs(self.player_x-int(self.wid / 2))<15:
            # print(self.player_x-int(self.wid/2))
            return "center"
        elif (self.player_x-int(self.wid / 2))<120 and (self.player_x-int(self.wid / 2))>65:
            return "left"
        elif (int(self.wid / 2)-self.player_x)<120 and (int(self.wid / 2)-self.player_x)>65:
            return "right"
        else:
            return None




# if __name__ == "__main__":
#     pc = PyCar()
#     pc.reset_game()
#     pc.run_game()