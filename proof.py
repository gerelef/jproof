#!/usr/bin/env python3.12
import json
import os
from sys import argv
from typing import TextIO


def auto_str(cls):
    """Automatically implements __str__ for any class."""

    def __str__(self):
        return '%s(%s)' % (
            type(self).__name__,
            ', '.join('%s=%s' % item for item in vars(self).items())
        )

    cls.__str__ = __str__
    return cls


def is_primitive(v):
    return isinstance(v, str) or isinstance(v, int) or isinstance(v, float) or isinstance(v, bool) or v is None


def is_container(v):
    return isinstance(v, list) or isinstance(v, dict)


@auto_str
class TrackedObject:
    def __init__(self, path="/"):
        self.path = path
        self.value_list = []
        self.conflict_count = 0  # existing field counter
        self.endorse_count = 0  # new field counter

    def increment(self, v):
        if isinstance(v, dict):
            for k, v in v.items():
                self.endorse(k, v)
            return

        self.endorse_count += 1
        self.value_list.append(v)

    def endorse(self, k, v):
        if hasattr(self, k):
            getattr(self, k).increment(v)
            self.conflict_count += 1
            return

        self.endorse_count += 1
        setattr(self, k, TrackedObject(os.path.join(self.path, k)))
        getattr(self, k).increment(v)


GLOBAL_MODEL: TrackedObject = TrackedObject()


def handle_json_dump_object(o: dict) -> None:
    global GLOBAL_MODEL
    for k, v in o.items():
        GLOBAL_MODEL.endorse(k, v)


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
    # TODO add --from-dump <input-file>
    # TODO add --from-array <input-file>
    with open(argv[1], "r") as fj_dump:
        while jobj := read_jobj_incrementally(fj_dump):
            handle_json_dump_object(json.loads(jobj))

    print(GLOBAL_MODEL)
