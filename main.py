import gym
import numpy as np
from drawer import Drawer, MOVE_TO, LINE_TO, CURVE_TO
import cv2

spaces = gym.spaces

action_space = spaces.Tuple([
    spaces.Discrete(3),
    # to point, control point1, control point2,
    spaces.Box(low=0, high=1, shape=(3, 2)),
    # control point 1
    # spaces.Box(low=0, high=1, shape=(1, 2)),
    # # control point 2
    # spaces.Box(low=0, high=1, shape=(1, 2)),
])


class AutoAodrawEnv(gym.Env):
    drawer: Drawer

    def __init__(self, width, height):
        super(AutoAodrawEnv, self).__init__()
        self.width = width
        self.height = height
        self.action_space = action_space
        self.observation_space = spaces.Box(low=0, high=255, shape=(width * 2, height), dtype=np.int)
        self.reset()

    def step(self, action: spaces.Discrete) -> (spaces.Box, np.float, bool, object):
        action_type = action[0]

        if self.drawer.draw_count == 0 and action_type != MOVE_TO:
            return self.get_obs(), -1, True, None
        self.drawer.draw(action)
        reward = self.get_reward()
        return self.get_obs(), reward, False, None

    def reset(self):
        self.drawer = Drawer(self.width, self.height)

    def render(self, mode='human'):
        pass

    def get_obs(self):
        pass

    @staticmethod
    def get_reward() -> int:
        return 1


def random_images():
    for i in range(0, 100):
        a = AutoAodrawEnv(200, 200)
        first_action = a.action_space.sample()
        while first_action[0] != MOVE_TO:
            first_action = a.action_space.sample()
        a.step(first_action)
        for j in range(50):
            act = a.action_space.sample()
            a.step(act)
        IMG_PATH = "images/" + str(i) + ".png"
        a.drawer.save_png(IMG_PATH)
        img = cv2.imread(IMG_PATH, 0)
        ret1, gray = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        cv2.imwrite(IMG_PATH, gray)


if __name__ == '__main__':
    random_images()

