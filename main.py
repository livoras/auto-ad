import gym
import numpy as np
from drawer import Drawer, MOVE_TO, LINE_TO, CURVE_TO, get_action_type
import cv2
import cairo

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines.common.env_checker import check_env

spaces = gym.spaces

action_space = spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float)


class AutoAodrawEnv(gym.Env):
    drawer: Drawer
    target_image_state: np.array

    def __init__(self, img: str):
        super(AutoAodrawEnv, self).__init__()
        img = cv2.imread(img)
        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        self.img = img
        width, height, channels = img.shape
        self.width = width
        self.height = height
        self.action_space = action_space
        self.observation_space = spaces.Box(low=0, high=255, shape=(width * height * 2,), dtype=np.int)
        self.reset()

    def step(self, action: spaces.Discrete) -> (spaces.Box, np.float, bool, object):
        action_type = get_action_type(action[0])

        if self.drawer.draw_count == 0 and action_type != MOVE_TO:
            return self.get_obs(), -1, True, {}
        self.drawer.draw(action)
        reward, done = self.get_reward()
        return self.get_obs(), reward, False if (reward >= 0 and not done) else True, {}

    def reset(self):
        self.drawer = Drawer(self.width, self.height)
        self.init_img_data()
        return self.get_obs()

    def render(self, mode='human'):
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.width, self.height * 2)
        data = surface.get_data()
        concatenated_image_data = self.get_obs()
        i = 0
        j = 0
        while i < len(data):
            val = concatenated_image_data[j]
            data[i] = val
            data[i + 1] = val
            data[i + 2] = val
            data[i + 3] = 255
            i += 4
            j += 1
        surface.write_to_png('result.png')

    def get_obs(self):
        return np.array(self.target_image_state + self.drawer.state)

    def get_reward(self) -> (float, bool):
        target_state = self.target_image_state
        current_state = self.drawer.state
        i = 0
        target_black_count = 0
        draw_success_count = 0
        while i < len(target_state):
            is_target_black = target_state[i] == 0
            is_current_black = current_state[i] == 0

            if is_target_black:
                target_black_count += 1

            # if drawing over white
            if not is_target_black and is_current_black:
                return -1, True

            # drawing good
            if is_target_black and is_current_black:
                draw_success_count += 1

            i += 1

        return (0, False) if target_black_count == 0 else (
            (draw_success_count / target_black_count) / self.drawer.draw_count,
            draw_success_count == target_black_count,
        )

    def init_img_data(self):
        img = self.img
        ret = []
        for x in range(self.width):
            for y in range(self.height):
                val = img[x, y, 0]
                ret.append(val)
        self.target_image_state = ret


def random_images():
    for i in range(0, 100):
        print(' --- ', i)
        a = AutoAodrawEnv('ojbk.png')
        first_action = a.action_space.sample()
        while get_action_type(first_action[0]) != MOVE_TO:
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


def learn():
    env = AutoAodrawEnv("images/0.png")
    env = DummyVecEnv([lambda: env])
    # action = env.action_space.sample()
    # obs, reward, done, _ = env.step(action)

    model = PPO2(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=10000)

    env.reset()
    for i in range(100):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        if done:
            env.render()
            break

    # print('action -> ', action)
    # print('obs -> ', obs)
    # print('reward -> ', reward)
    # print('done -> ', done)


if __name__ == '__main__':
    # env = AutoAodrawEnv("images/0.png")
    # check_env(env)
    learn()
    # random_images()
