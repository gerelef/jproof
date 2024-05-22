#!/usr/bin/env python3.12
import json
from collections import UserDict, UserList
from sys import argv
from typing import Self, TextIO

type TrackedDict = ...
type TrackedArray = ...
type TrackedPrimitive = ...


def transform_json_to_tracked[T](thing: T) -> TrackedDict | TrackedArray | TrackedPrimitive:
    if isinstance(thing, dict):
        obj = TrackedDict()
        for k, v in thing.items():
            obj[k] = v
        return obj

    if isinstance(thing, list):
        arr = TrackedArray()
        arr.append(thing)
        return arr

    return TrackedPrimitive(thing)


class TrackedPrimitive:
    def __init__(self, value):
        self.instances = 1
        self.values = [value]

    def __str__(self):
        return f"TrackedPrimitive {self.values}"

    def __repr__(self):
        return str(self)

    @property
    def data(self) -> ...:
        return self.values[-1]

    def increment(self, key, value) -> Self:
        self.instances += 1
        self.values.append(value)
        return self


class TrackedArray(UserList):
    def __str__(self):
        return f"TrackedArray {self.data}"

    def __repr__(self):
        return str(self)

    def increment(self, key, value) -> Self:
        self.append(value)
        return self


class TrackedDict(UserDict):
    def __init__(self):
        super().__init__()
        self.total_additions = 0
        self.total_overwrites = 0

    def __setitem__(self, key, value):
        value = transform_json_to_tracked(value)
        if key not in self.data:
            self.total_additions += 1
            self.data[key] = value
            return

        # if this is a tracked entry, increment by the new value
        self.total_overwrites += 1
        self.data[key].increment(key, value)

    def __str__(self) -> str:
        return f"TrackedDict {self.data}"

    def __repr__(self):
        return str(self)

    def increment(self, key, value) -> Self:
        self[key] = value
        return self


GLOBAL_MODEL: TrackedDict[str, ...] = TrackedDict()


def handle_json_dump_object(o: dict) -> None:
    global GLOBAL_MODEL
    for k, v in o.items():
        GLOBAL_MODEL[k] = v


def read_jobj_incrementally(f: TextIO) -> str | None:
    """
    :return: a string representation of a json object from file
    """
    mark = f.tell()
    total_bytes_read = 0

    def get_next_tokens() -> str:
        nonlocal total_bytes_read, offset
        buffer_size = 4096
        offset = buffer_size - 1
        total_bytes_read += buffer_size
        return f.read(buffer_size)

    entity = ""
    offset = 0
    quoted = False
    # the first buffer's whitespace to the left is insignificant
    buffer = get_next_tokens().lstrip()
    if not buffer:
        return None
    bracket_count = 0
    assert buffer[0] == "{"
    while True:
        previous = None
        for c in buffer:
            offset -= 1
            entity += c
            if c == "\"" and previous != "\\":
                quoted = not quoted
            bracket_offset = 0
            if not quoted and c == "{":
                bracket_offset = 1
            if not quoted and c == "}":
                bracket_offset = -1

            bracket_count += bracket_offset
            if bracket_count == 0:
                break
            previous = c

        if bracket_count == 0:
            break
        buffer = get_next_tokens()
    f.seek(mark + total_bytes_read - offset)
    return entity


if __name__ == "__main__":
    with open(argv[1], "r") as fj_dump:
        while jobj := read_jobj_incrementally(fj_dump):
            handle_json_dump_object(json.loads(jobj))

    print(GLOBAL_MODEL)
    print(GLOBAL_MODEL.total_additions)
    print(GLOBAL_MODEL.total_overwrites)
