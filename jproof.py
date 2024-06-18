#!/usr/bin/env python3.12
import json
from sys import argv
from types import NoneType
from typing import TextIO, Any, Self

type Key = str
type JsonArray = list
type JsonObject = dict
type Composite = JsonObject | JsonArray
type Primitive = str | int | float | bool | None


def auto_str(cls):
    """Automatically implements __str__ for any class."""

    def __str__(self):
        return '%s%s(%s)' % (
            type(self).__name__,
            f"@{id(self)}",
            f','.join('%s=%s' % item for item in vars(self).items())
        )

    cls.__str__ = __str__
    cls.__repr__ = __str__
    return cls


@auto_str
class Node:
    def __init__(self, path: list = None):
        self.path = path if path else []
        self.keyed_data: dict[Key, list[Primitive | Composite]] = {}
        self.total_endorsements = 0

    def _create_node(self, k: Key, o: ...) -> Self | ...:
        if not isinstance(o, dict):
            return o

        o: dict
        jobj = Node([*self.path, k])
        jobj.endorse_jobj(o)
        return jobj

    def endorse_jobj(self, jobj: JsonObject):
        self.total_endorsements += 1
        for k, v in jobj.items():
            if k not in self.keyed_data:
                self.keyed_data[k] = [self._create_node(k, v)]
                continue

            self.keyed_data[k].append(self._create_node(k, v))

    def _aggragate_field_types(self) -> dict[str, str | list[str] | dict]:
        schema = {"type": translate_to_primitive_schema_type(self), "properties": {}}
        property_types = {}
        for key, values in self.keyed_data.items():
            property_types[key] = []
            values: list
            for item in values:
                property_types[key].append(translate_to_primitive_schema_type(item))

        if len(property_types.keys()) > 0:
            for key, _ in self.keyed_data.items():
                schema["properties"] = schema["properties"] | {key: {"type": list(set(property_types[key]))}}
        return schema

    def schema(self) -> dict:
        return self._aggragate_field_types()


def translate_to_primitive_schema_type(obj: Any | type | None) -> str:
    if obj is None or obj is type(NoneType):
        return "null"
    if isinstance(obj, str) or obj is str:
        return "string"
    if isinstance(obj, float) or obj is float:
        return "number"
    if isinstance(obj, int) or obj is int:
        return "integer"
    if isinstance(obj, dict) or isinstance(obj, Node) or obj is dict:
        return "object"
    if isinstance(obj, list) or obj is list:
        return "array"
    if isinstance(obj, bool) or obj is bool:
        return "boolean"
    raise RuntimeError(obj)


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
    model: Node = Node()  # root model
    with open(argv[1], "r") as fj_dump:
        while jobj := read_jobj_incrementally(fj_dump):
            model.endorse_jobj(json.loads(jobj))

    print(model)
    print(model.schema())
