import cairo
from gym import spaces

MOVE_TO = 0
LINE_TO = 1
CURVE_TO = 2


def grey_sale(nums):
    ret = []
    for num in nums:
        ret.append(255 if num > 127 else 0)


class Drawer:
    surface: cairo.ImageSurface
    draw_count: int = 0
    width: int
    height: int

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

    def draw(self, action: [spaces.Discrete, spaces.Box]):
        draw_type = action[0]
        pts = action[1]
        to_point = self.to_xy(pts[0])
        ctx = self.ctx

        if draw_type is MOVE_TO:
            ctx.move_to(to_point[0], to_point[1])
        elif draw_type is LINE_TO:
            ctx.line_to(to_point[0], to_point[1])
        elif draw_type is CURVE_TO:
            ctr1 = self.to_xy(pts[1])
            ctr2 = self.to_xy(pts[2])
            ctx.curve_to(ctr1[0], ctr1[1], ctr2[0], ctr2[1], to_point[0], to_point[1])
        else:
            raise Exception("Not support render type " + str(draw_type))

        ctx.stroke()
        self.draw_count += 1

    def get_rgba(self):
        i = 0
        content = self.surface.get_data()
        rgba = []
        while i < len(content):
            rgba.append(grey_sale((content[i], content[i + 1], content[i + 2], content[i + 3])))
            i += 4
        return rgba

    def save_png(self, path: str):
        self.surface.write_to_png(path)

    def to_xy(self, action: [float, float]) -> (int, int):
        return self.width * action[0], self.height * action[1]
