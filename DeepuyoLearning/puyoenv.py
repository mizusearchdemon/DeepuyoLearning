import random
import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import math

#import cupy as np

class PuyoEnv(gym.Env):

    def __init__(self, seed = None, field_width = 6, field_height = 14, num_of_colors = 4, field = None):
        self.action_space = spaces.Discrete(22)
        if field is None:
            self.field = np.zeros(dtype=np.uint8, shape=(field_width, field_height,))
        else:
            self.field = field

        self.num_of_colors = num_of_colors
        self.field_width  = self.field.shape[0]
        self.field_height = self.field.shape[1]
        self.num_of_visible_tsumo = 3
        self.reset()

        if field is not None:
            self.field = field
            self.fall_puyo()

        self.viewer = None
        self.action_space = spaces.Discrete(self.field_width * 4 - 2)
        self.action_space_to_line_rotate = []
        for i in range(self.action_space.n + 2):
            if (i // 4 == 0 and i % 4 == 3): continue
            if (i // 4 == (self.field_width - 1) and i % 4 == 1): continue
            self.action_space_to_line_rotate.append((i // 4, i % 4))
        # 色数が足りなければ増やすこと
        self.colors = np.array([
            (1, 1, 1), # empty
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 1, 0),
            (1, 0, 1),
            (0, 1, 1),
            (.5, 0, 0),
            (0, .5, 0),
            (0, 0, .5),
            (.5, .5, 0),
            (.5, 0, .5),
            (0, .5, .5),
        ], dtype=np.float)

    def set_seed(self, seed = None):
        if seed is None:
            seed = random.SystemRandom().randint(0, 0xFFFFFFFF)
        np.random.seed(seed)
        self.seed = seed
        return seed

    def make_tsumo(self):
        self.length_of_tsumo_list = 1000
        self.tsumo_list = np.random.randint(1, self.num_of_colors + 1, (self.length_of_tsumo_list, 2))
        return

    def as2lr(self, actions):
        return [self.action_space_to_line_rotate[a] for a in actions]

    def get_next_tsumo(self):
        if self.tsumo_list is None:
            make_tsumo()

        if self.next_visible_tsumo_i >= self.length_of_tsumo_list:
            make_tsumo()
            #self.next_visible_tsumo_i = 0

        for i in range(self.num_of_visible_tsumo - 1):
            self.current_tsumo[i] = self.current_tsumo[i + 1]
        self.current_tsumo[self.num_of_visible_tsumo - 1] = self.tsumo_list[self.next_visible_tsumo_i % self.length_of_tsumo_list]
        self.next_visible_tsumo_i += 1
        return

    def make_observation(self):
        _field = np.reshape(self.field, (1, -1))
        _tsumo = np.reshape(self.current_tsumo, (1, -1))
        self.observation_space = np.c_[_field, _tsumo]
        return self.observation_space

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        line, rotate = self.action_space_to_line_rotate[action]
        field = self.field
        _reachable_height = self.get_reachable_height()

        reward = 0
        done = False

        # ぷよ置けるか判断
        # rotateは、軸ぷよを基準にもう1個のぷよの位置が↑の場合0で、以降は時計回り 0:12時 1:3時 2:6時 3:9時
        valid_action = True
        #if line == 0 and rotate == 3: valid_action = False   # 左端の列に、左に倒して置けない
        #elif line == 5 and rotate == 1: valid_action = False # 右端の列に、右に倒して置けない
        if field[line][self.field_height - 2] > 0:
            valid_action = False # 13段目が埋まっていたら置けない
        #elif self.height[line] >= (self.field_height - 2) and rotate == 2:
            # 14段目には軸ぷよを持っていけない
            #valid_action = False
        elif _reachable_height[line] == 0:
            # 列に辿り着けない
            valid_action = False
        elif rotate == 1 and _reachable_height[line + 1] == 0:
            # 右の列に置けない
            valid_action = False
        elif rotate == 3 and _reachable_height[line - 1] == 0:
            # 左の列に置けない
            valid_action = False

        if valid_action:
            # ぷよを置く
            self.put_current_tsumo(line, rotate)
            # 連鎖判定
            rensa, point = self.exec_rensa()
            reward = rensa ** 2

            if self.height[2] >= self.field_height - 2:
                # 窒息
                valid_action = False
            elif rensa >= 14:
                done = True
                #reward = rensa ** 2
            elif self.next_visible_tsumo_i > 100:
                valid_action = False

        obs = self.make_observation()

        if not valid_action:
            return obs, -1, True, {}

        return obs, reward, done, {}

    def reset(self, seed = None):
        self.field = np.zeros_like(self.field)
        self.height = np.zeros(dtype=np.uint8, shape=(self.field_width,))

        self.set_seed(seed)
        self.current_tsumo = np.zeros(dtype=np.uint8, shape=(self.num_of_visible_tsumo, 2))
        self.next_visible_tsumo_i = 0
        self.make_tsumo()
        for i in range(self.num_of_visible_tsumo):
            self.get_next_tsumo()

        self.steps_beyond_done = None
        obs = self.make_observation()
        return obs

    def _rect_geom(self, l, r, t, b):
        return [(l,b), (l,t), (r,t), (r,b)]

    def _draw_puyos(self, viewer, puyos, oxoy, puyo_size):
        for col, rows in enumerate(puyos):
            for row, color in enumerate(rows):
                l,t = oxoy[0] + puyo_size * col, oxoy[1] + puyo_size * row
                r,b = l + puyo_size, t + puyo_size
                self.viewer.draw_polygon(self._rect_geom(l,r,t,b), color=self.colors[color])

    def render(self, mode='human'):
        field_col = self.field_width
        field_row = self.field_height
        field = self.field
        current_tsumo = self.current_tsumo
        # print('field:',field)
        # print('tsumo:',current_tsumo)

        screen_width = 600
        screen_height = 600
        # ツモ表示エリアが上にあって、その下にフィールドがある
        tsumo_area_height = 40
        scale = min(
            screen_width / self.field_width,
            screen_width / (self.field_height + tsumo_area_height)
        )
        puyo_size = math.floor(scale)

        tsumo_ox = (screen_width - field_col * puyo_size) / 2
        tsumo_oy = (screen_height - field_row * puyo_size) / 2
        field_ox = tsumo_ox
        field_oy = tsumo_oy + tsumo_area_height
        field_width = field_col * puyo_size
        field_height = field_row * puyo_size
        tsumo_area_width = field_width
        # tsumo_area_height = 定義済み

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

        # tsumo puyo
        l,r,t,b = tsumo_ox, tsumo_ox + tsumo_area_width, tsumo_oy, tsumo_oy + tsumo_area_height
        self.viewer.draw_polygon(self._rect_geom(l,r,t,b), filled=True, color=(1,1,1))
        for i, tsumo in enumerate(current_tsumo):
          space = 5
          size = min(puyo_size, ((tsumo_area_height - space * 2) / 2))
          oxoy = tsumo_ox + (size + space) * i, tsumo_oy + space
          self._draw_puyos(self.viewer, np.reshape(tsumo, (1,2)), oxoy, size)

        # field puyo
        self._draw_puyos(self.viewer, field, [field_ox, field_oy], puyo_size)

        # draw field
        field_color = (.5,.5,.5)
        l,r,t,b = field_ox, field_ox + field_width, field_oy, field_oy + field_height
        #self.viewer.draw_polyline(self._rect_geom(l,r,t,b), color=field_color)
        self.viewer.draw_polygon(self._rect_geom(l,r,t,b), filled=False, color=field_color)
        # 縦線
        for c in range(field_col):
            x = field_ox + c * puyo_size
            self.viewer.draw_polyline([(x, field_oy), (x, field_oy + field_height)], color=field_color)
        # 横線
        for r in range(field_row):
            y = field_oy + r * puyo_size
            self.viewer.draw_polyline([(field_ox, y), (field_ox + field_width, y)], color=field_color)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer: self.viewer.close()

    def get_reachable_height(self, current_line = 2, current_height = 11):
        height = self.height
        l = r = pivot = current_line
        highest = current_height
        while True:
            prev_highest = highest

            # 左方向
            for w in range(pivot,-1,-1):
                if height[w] > highest:
                    break # それ以上いけない
                if height[w] == highest and highest < self.field_height: # - 1: # 14段目に持っていけないときは-1
                    highest += 1  # 回しで一段上れる
                    break
                l = w
            if prev_highest != highest:
                continue

            # 右方向
            for w in range(pivot,self.field_width):
                if height[w] > highest:
                    break # それ以上いけない
                if height[w] == highest and highest < self.field_height - 1:
                    highest += 1  # 回しで一段上れる
                    break
                r = w

            if prev_highest == highest:
                break

        reachable_height = np.zeros_like(self.height)
        for w in range(l, r+1):
            reachable_height[w] = highest

        return reachable_height

    def put_current_tsumo(self, line, rotate):
        for i in range(2):
            if self.height[line] < self.field_height - 1:
                self.field[line][self.height[line]] = self.current_tsumo[0][0]
                self.height[line] += 1
            if rotate == 1: line += 1
            elif rotate == 3: line -= 1

        self.get_next_tsumo()
        return

    def delete_puyo(self, w, h):
        color = self.field[w][h]
        self.field[w][h] = 0
        n = 1
        if h < self.field_height - 2 and self.field[w][h+1] == color: n += self.delete_puyo(w, h+1)
        if h > 0                     and self.field[w][h-1] == color: n += self.delete_puyo(w, h-1)
        if w < self.field_width - 1  and self.field[w+1][h] == color: n += self.delete_puyo(w+1, h)
        if w > 0                     and self.field[w-1][h] == color: n += self.delete_puyo(w-1, h)
        return n

    def fall_puyo(self):
        for w in range(self.field_width):
            h_blank = 0
            self.field[w][self.field_height-1] = 0 # 最上段は消す
            for h in range(self.field_height - 1): # 最上段は落とさない
                if self.field[w][h_blank] > 0:
                    h_blank = h + 1
                elif self.field[w][h_blank] == 0 and self.field[w][h] > 0:
                    self.field[w][h_blank] = self.field[w][h]
                    self.field[w][h] = 0
                    h_blank += 1
            self.height[w] = h_blank
        return

    def exec_rensa(self):
        rensa = 0
        point = 0

        field = self.field
        field_width = self.field_width
        field_height = self.field_height
        
        self.fall_puyo()

        while True:
            #print(self.field)
            connection_count = np.zeros_like(field)
            type_of_delete_puyo_colors = {}
            combination_bonus = 0
            total_num_of_delete_puyos = 0

            # 隣り合う色と一致する数
            for w in range(field_width):
                for h in range(field_height - 2):
                    if field[w][h] == 0: continue # ぷよが無い
                    if h < field_height - 3 and field[w][h] == field[w][h+1]: connection_count[w][h] += 1
                    if h > 0                and field[w][h] == field[w][h-1]: connection_count[w][h] += 1
                    if w < field_width - 1  and field[w][h] == field[w+1][h]: connection_count[w][h] += 1
                    if w > 0                and field[w][h] == field[w-1][h]: connection_count[w][h] += 1

            # 隣同士と一致した数が3以上 または 2以上が隣り合ってる 場合は消す
            for w in range(field_width):
                for h in range(field_height - 2):
                    if (field[w][h] == 0): continue # ぷよが無い

                    num_of_delete_puyos = 0
                    color = field[w][h]
                    if (connection_count[w][h] <= 1): continue
                    elif (connection_count[w][h] >= 3):
                        num_of_delete_puyos = self.delete_puyo(w, h)
                    else:
                        if   h < field_height - 3 and field[w][h] == field[w][h+1] and connection_count[w][h+1] >= 2: num_of_delete_puyos = self.delete_puyo(w, h)
                        elif h > 0                and field[w][h] == field[w][h-1] and connection_count[w][h-1] >= 2: num_of_delete_puyos = self.delete_puyo(w, h)
                        elif w < field_width - 1  and field[w][h] == field[w+1][h] and connection_count[w+1][h] >= 2: num_of_delete_puyos = self.delete_puyo(w, h)
                        elif w > 0                and field[w][h] == field[w-1][h] and connection_count[w-1][h] >= 2: num_of_delete_puyos = self.delete_puyo(w, h)

                    if (num_of_delete_puyos > 0): # ぷよを消した
                        total_num_of_delete_puyos += num_of_delete_puyos
                        type_of_delete_puyo_colors[color] = True
                        if num_of_delete_puyos >= 5:
                            combination_bonus += (2, 3, 4, 5, 6, 7, 10)[min(num_of_delete_puyos, 11) - 5]

            if total_num_of_delete_puyos == 0:
                break

            rensa_bonus = (0, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512)[rensa]
            color_bonus = (0, 3, 6, 12, 24)[len(type_of_delete_puyo_colors) -1]
            point += total_num_of_delete_puyos * 10 * max((rensa_bonus + combination_bonus + color_bonus), 1)

            # 浮いたぷよを落下
            self.fall_puyo()

            rensa += 1
            #print("rensa = ", rensa, "ppint = ", point)

        return (rensa, point)

    def field_to_url(self):
        url = "http://www.inosendo.com/puyo/rensim/??"
        color_to_url = {0: 0, 1: 4, 2: 7, 3: 5, 4: 6, 5: 8, 10: 1}

        for h in range(self.field_height - 1)[::-1]:
            for w in range(self.field_width)[::-1]:
                url += str(color_to_url[self.field[w][h]])

        return url

    def url_to_field(self, url):
        puyos = list(url.split("??")[1])
        url_to_color = {'0': 0, '4': 1, '7': 2, '5': 3, '6': 4, '8': 5, '1': 10}

        for h in range(self.field_height - 1)[::-1]:
            for w in range(self.field_width)[::-1]:
                self.field[w][h] = url_to_color[puyos.pop(0)]

        return self.field
