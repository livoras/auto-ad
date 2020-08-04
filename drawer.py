import cairo
from gym import spaces
import numpy as np
import math

MOVE_TO = 0
LINE_TO = 1
CURVE_TO = 2


def grey_sale(nums):
    ret = []
    for num in nums:
        ret.append(255 if num > 127 else 0)
    return ret


def get_action_type(n):
    try:
        return int(round(n * 100) % 3)
    except ValueError:
        print("~~~~~~~~~~~~~~~", n, "~~~~~~~~~~~~~~")


def map_to_01(n):
    return n


class Drawer:
    surface: cairo.ImageSurface
    draw_count: int = 0
    width: int
    height: int
    state: np.array

    def __init__(self, width: int, height: int):

        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        ctx = cairo.Context(surface)
        self.surface = surface
        self.ctx = ctx
        self.width = width
        self.height = height

        ctx.set_source_rgba(255, 255, 255, 1)
        ctx.rectangle(0, 0, width, height)
        ctx.fill()
        ctx.set_source_rgba(0, 0, 0, 1)

        self.state = self.get_image_data()

    def draw(self, action: [spaces.Discrete, spaces.Box]):
        draw_type = get_action_type(action[0])
        to_point = self.to_xy(action[1], action[2])
        ctx = self.ctx

        if draw_type == MOVE_TO:
            ctx.move_to(to_point[0], to_point[1])
        elif draw_type == LINE_TO:
            ctx.line_to(to_point[0], to_point[1])
        elif draw_type == CURVE_TO:
            ctr1 = self.to_xy(action[3], action[4])
            ctr2 = self.to_xy(action[5], action[6])
            # print(ctr1, ctr2)
            ctx.curve_to(ctr1[0], ctr1[1], ctr2[0], ctr2[1], to_point[0], to_point[1])
        else:
            print('THIS IS THE ACTION', action)
            raise Exception("Not support render type " + str(draw_type))

        ctx.stroke()
        self.draw_count += 1
        self.state = self.get_image_data()

    def get_image_data(self):
        i = 0
        content = self.surface.get_data()
        rgba = []
        j = 0
        while i < len(content):
            # rgba.append(grey_sale((content[i], content[i + 1], content[i + 2], content[i + 3])))
            j += 1
            val = grey_sale([content[i]])
            rgba.append(val[0])
            i += 4
        return rgba

    def save_png(self, path: str):
        self.surface.write_to_png(path)

    def to_xy(self, x, y) -> (int, int):
        return self.width * map_to_01(x), self.height * map_to_01(y)
