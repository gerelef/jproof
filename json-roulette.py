#!/usr/bin/env python3.12
import argparse
import json
import random
import string
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


def decide(percentage: float) -> bool:
    """
    :param percentage: percentage <= 1.0
    """
    assert percentage <= 1.0
    return round(random.uniform(0, 1), 3) <= percentage


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


def generate_random_string(nullable=False) -> str | None:
    # 1 in 10 chance return null
    if nullable and decide(0.1):
        return None
    return "".join(random.choices(string.ascii_letters, k=random.randint(1, 5)))


def generate_random_int(nullable=False) -> int | None:
    # 1 in 10 chance return null
    if nullable and decide(0.1):
        return None
    return random.randint(-999, 10000)


def generate_random_double(nullable=False) -> float | None:
    # 1 in 10 chance return null
    if nullable and decide(0.1):
        return None
    return random.uniform(-1000, 1000)


def generate_random_bool(nullable=False) -> bool | None:
    # 1 in 10 chance return null
    if nullable and decide(0.1):
        return None
    return decide(random.uniform(0.2, 0.8))


def generate_random_jfield(*primitive_generators: Callable[[], object]) -> tuple[str, ...]:
    return random.choice(words), random.choice(primitive_generators)()


def generate_jobj(
        current_depth: int,
        length_low: int,
        length_high: int,
        nested_chance: float,
        nested_max_depth: int) -> dict:
    out = {}
    jobj_length = random.randint(length_low, length_high)
    for i in range(jobj_length):
        key, value = generate_random_jfield(
            generate_random_string,
            generate_random_int,
            generate_random_double,
            generate_random_bool
        )
        if current_depth < nested_max_depth and decide(nested_chance):
            key = random.choice(words)
            value = generate_jobj(
                current_depth + 1,
                length_low,
                length_high,
                nested_chance,
                nested_max_depth
            )
        out[key] = value
    return out


def generate_jarr(
        current_depth: int,
        length_low: int,
        length_high: int,
        nested_chance: float,
        nested_max_depth: int) -> dict:
    raise NotImplementedError()


@dataclass
class UserOptions:
    # --size
    output_size: int
    composites_size_low: int
    composites_size_high: int
    # --object
    # --array
    output_is_jobject_else_array: bool  # True means generate json objects, otherwise generate arrays
    # --word-file
    path_to_word_file: Path
    # --word-sample-size
    word_sample_size: int
    # --nested-chance
    # --flat is this thing but 0.0
    nested_chance: float
    # --nested-max-depth
    nested_max_depth: int
    # --pretty
    pretty: bool


def _parse_args(args) -> UserOptions:
    parser = argparse.ArgumentParser(description="json-roulette: a barebones json generator, for testing")
    parser.add_argument("--size", type=int, required=True)
    # group = parser.add_mutually_exclusive_group(required=True)
    # disabled until further notice
    # group.add_argument("--objects", default=False, action="store_true")
    # group.add_argument("--arrays", default=False, action="store_true")
    parser.add_argument("--composites-size-low", type=int, required=True)
    parser.add_argument("--composites-size-high", type=int, required=True)
    # optionals
    parser.add_argument("--word-file", type=str, default="./resources/words", required=False)
    parser.add_argument("--word-sample-size", type=int, default=50, required=False)
    nested_flags_group = parser.add_mutually_exclusive_group(required=False)
    nested_flags_group.add_argument("--flat", default=False, action="store_true", required=False)
    nested_flags_group.add_argument("--nested-chance", type=float, default=0.2, required=False)
    parser.add_argument("--nested-max-depth", type=int, default=9999, required=False)
    parser.add_argument("--pretty", default=False, action="store_true", required=False)
    options = parser.parse_args(args)
    assert 1 <= options.size
    assert 1 <= options.composites_size_low <= options.composites_size_high
    assert 1 <= options.word_sample_size
    assert 0.0 <= options.nested_chance <= 1.0
    assert 1 <= options.nested_max_depth
    return UserOptions(
        output_is_jobject_else_array=True,
        # disabled until further notice
        # output_is_jobject_else_array=options.objects,
        output_size=options.size,
        composites_size_low=options.composites_size_low,
        composites_size_high=options.composites_size_high,
        path_to_word_file=Path(options.word_file).expanduser().absolute(),
        word_sample_size=options.word_sample_size,
        nested_chance=-1 if options.flat else options.nested_chance,
        nested_max_depth=1 if options.flat else options.nested_max_depth,
        pretty=options.pretty
    )


words = []
if __name__ == "__main__":
    random.seed(1337)
    sys.setrecursionlimit(100_000)
    options = _parse_args(sys.argv[1:])
    with open(options.path_to_word_file, "r") as dictionary:
        for line in dictionary:
            words.append(line.strip())
        words = tuple(sorted(random.choices(words, k=options.word_sample_size)))

    for i in range(options.output_size):
        generated = generate_jobj(
            0,
            options.composites_size_low,
            options.composites_size_high,
            options.nested_chance,
            options.nested_max_depth
        ) if options.output_is_jobject_else_array else generate_jarr(
            0,
            options.composites_size_low,
            options.composites_size_high,
            options.nested_chance,
            options.nested_max_depth
        )
        print(json.dumps(generated, indent=4 if options.pretty else None))
