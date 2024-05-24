#!/usr/bin/env python3.12
import json
import random
import string
import sys
from functools import partial
from typing import Callable


def field_roulette(json: dict) -> bool:
    """
    Do nothing, remove any field, or return True to add a field.
    :return: whether to add a field.
    """
    edit_field = random.randint(1, 15) == 1
    remove_field = random.randint(1, 2) == 1
    if edit_field and remove_field:
        del json[random.choice(list(json.keys()))]
        return False

    return edit_field


def generate_random_string(lower: int | None, upper: int) -> str | None:
    # if lower bound is None, w/ a 1 in 10 chance return null
    if not lower and random.randint(1, 25) == 1:
        return None
    lower = lower if lower else 0
    return "".join(random.choices(string.ascii_letters, k=random.randint(lower, upper)))


def generate_random_int(lower: int | None, upper: int) -> int | None:
    # if lower bound is None, w/ a 1 in 10 chance return null
    if not lower and random.randint(1, 25) == 1:
        return None
    lower = lower if lower else 0
    return random.randint(lower, upper)


def generate_random_double(lower: float | None, upper: float) -> float | None:
    # if lower bound is None, w/ a 1 in 10 chance return null
    if not lower and random.randint(1, 25) == 1:
        return None
    lower = lower if lower else 0
    return random.uniform(lower, upper)


def generate_random_bool(nullable: bool | None) -> bool | None:
    if (nullable is None or nullable) and random.randint(1, 25) == 1:
        return None
    return random.randint(1, 2) == 1


def create_random_array(uniform: bool | None, lower: int | None, upper: int) -> list[...] | None:
    uniform = uniform if uniform is not None else generate_random_bool(False)
    if not lower and random.randint(1, 25) == 1:
        return None
    generators = [random.choice(item_generators)] if uniform else random.choices(item_generators, k=random.randint(2,
                                                                                                                   len(item_generators)))
    lower = lower if lower else 0
    final_length = random.randint(lower, upper)
    return [random.choice(generators)() for _ in range(final_length)]


def create_random_object(lower: int | None, upper: int) -> dict | None:
    if not lower and random.randint(1, 25) == 1:
        return None
    generators = random.choices(item_generators, k=random.randint(1, len(item_generators)))
    lower = lower if lower else 0
    final_length = random.randint(lower, upper)
    keys = []
    for _ in range(final_length):
        keys.append(generate_random_string(1, 6))
    out = {k: random.choice(generators)() for k in keys}
    for _ in range(final_length):
        if field_roulette(out):
            out[generate_random_string(3, 7)] = random.choice(generators)()
    return out


item_generators: list[Callable] = [
    partial(generate_random_bool, None),
    partial(generate_random_double, None, 1000),
    partial(generate_random_int, None, 100),
    partial(generate_random_string, None, 35),
    partial(create_random_array, None, None, random.randint(20, 45)),
    partial(create_random_object, None, random.randint(1, 3))
]

sys.setrecursionlimit(100_000)

for _ in range(500):
    # bored, don't want to fix
    try:
        print(json.dumps(create_random_object(2, 4)))
    except RecursionError as ignored:
        continue
