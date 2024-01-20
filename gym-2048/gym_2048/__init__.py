from gymnasium.envs.registration import register
from .env import Base2048Env

register(
    id='Tiny2048-v0',
    entry_point='gym_2048.env:Base2048Env',
    kwargs={
        'width': 2,
        'height': 2,
    }
)

register(
    id='Med2048-v0',
    entry_point='gym_2048.env:Base2048Env',
    kwargs={
        'width': 3,
        'height': 3,
    }
)


register(
    id='2048-v0',
    entry_point='gym_2048.env:Base2048Env',
    kwargs={
        'width': 4,
        'height': 4,
    }
)
