#!/usr/bin/env python3.12
import argparse
import enum
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO, Any, Self

type Key = str
type JsonPath = str
type JsonArray = list
type JsonObject = dict
type Composite = JsonObject | JsonArray
type Primitive = str | int | float | bool | None


def http_import(url, sha256sum) -> [object, str]:
    """
    Load single-file lib from the web.
    :returns: types.ModuleType, filename
    """

    class HashMismatchException(Exception):
        pass

    class NoSha256DigestProvided(Exception):
        pass

    if sha256sum is None:
        raise NoSha256DigestProvided()
    import os
    import types
    import hashlib
    import urllib.request
    import urllib.parse
    code = urllib.request.urlopen(url).read()
    digest = hashlib.sha256(code, usedforsecurity=True).hexdigest()
    if digest == sha256sum:
        filename = os.path.basename(urllib.parse.unquote(urllib.parse.urlparse(url).path))
        module = types.ModuleType(filename)
        exec(code, module.__dict__)
        return module, filename
    raise HashMismatchException(f"SHA256 DIGEST MISMATCH:\n\tEXPECTED: {sha256sum}\n\tACTUAL: {digest}")


utils, _ = http_import(
    "https://raw.githubusercontent.com/gerelef/dotfiles/main/scripts/utils/modules/helpers.py",
    "a3f50fac78f2dc71f5c4541f29837c8c8a7595190f3c328a6f26db6bd786b6f1"
)


class NodePathDoesNotExist(Exception):
    pass


@utils.auto_str
class Schema:
    @dataclass
    class Metadata:
        key: Key
        types: list[str] | str
        constraints: list | None
        frequency: float  # 0 < hz <= 1.0

    def __init__(self):
        pass


@utils.auto_str
class Node:
    PATH_SEPARATOR = "."

    def __init__(self, path: JsonPath = "$"):
        self.__path: JsonPath = path
        self.keyed_data: dict[Key, list[Primitive | Composite | Node]] = {}
        self.keyed_endorsements: dict[Key, int] = {}
        self.total_endorsements = 0

    @property
    def path(self) -> JsonPath:
        """
        :return: the current path
        """
        return self.__path

    @property
    def name(self) -> str:
        return self.path.split(Node.PATH_SEPARATOR)[-1]

    def abs_path(self, k: Key) -> JsonPath:
        """
        :return: the absolute path to a key
        """
        return Node.PATH_SEPARATOR.join([*self.path.split(Node.PATH_SEPARATOR), k])

    def rel_path(self, path: Key) -> JsonPath:
        """
        :return: the relative path to a key, from the current node
        """
        return path.removeprefix(self.path).removeprefix(Node.PATH_SEPARATOR)

    def _increment_endorsement(self, k: Key) -> Self:
        if k not in self.keyed_endorsements:
            self.keyed_endorsements[k] = 1
            return self

        self.keyed_endorsements[k] += 1
        return self

    def _create_node(self, k: Key, o: ...) -> Self | ...:
        if not isinstance(o, dict):
            return o

        o: dict
        jobj = Node(path=self.abs_path(k))
        jobj.endorse_jobj(o)
        return jobj

    def _aggragate_field_types(self,
                               inclusion_tolerance: float,
                               required_tolerance: float,
                               do_property_description_prompt: bool,
                               title: str | None = None,
                               description: str | None = None) -> dict[str, str | list[str] | dict]:
        schema = {"type": translate_to_primitive_schema_type(self)}
        property_types = {}
        for key, values in self.keyed_data.items():
            property_types[key] = []
            values: list
            for item in values:
                property_types[key].append(translate_to_primitive_schema_type(item))

        if len(property_types.keys()) > 0:
            schema["properties"] = {}
            for key, value in property_types.items():
                jtype = list(set(value))
                final_jtype = {"type": jtype}
                # TODO this is wrong on so many levels
                #  object definitions/properties can coexist with primitives, so this is redundant & incorrect
                field_has_own_properties = "object" in jtype if JOBJECT_SUPERSEDES_PRIMITIVES else (
                        len(jtype) == 1 and jtype[0] == "object"
                )
                if field_has_own_properties:
                    reference = None
                    for v in self.keyed_data[key]:
                        if isinstance(v, Node):
                            reference = v
                            break
                    final_jtype = reference.schema(
                        inclusion_tolerance,
                        required_tolerance,
                        do_property_description_prompt,
                        title,
                        description,
                    )

                schema["properties"] = schema["properties"] | {key: final_jtype}
        return schema

    def endorse_jobj(self, jobj: JsonObject | dict) -> Self:
        self.total_endorsements += 1
        for k, v in jobj.items():
            self._increment_endorsement(k)
            to_endorse = self._create_node(k, v)

            if k not in self.keyed_data:
                self.keyed_data[k] = [to_endorse]
                continue

            # special case
            # if we're about to endorse a node, check if it actually exists so no multiple nodes exist as values
            #  since composites can do their own tracking, we don't have to keep everything about them
            if isinstance(to_endorse, Node):
                endorsed_node = list(filter(lambda i: isinstance(i, Node), self.keyed_data[k]))
                assert len(endorsed_node) <= 1  # sanity
                if len(endorsed_node) == 1:
                    endorsed_node[0].endorse_jobj(v)
                    continue
            self.keyed_data[k].append(to_endorse)

        return self

    def get_property(self, path: JsonPath) -> tuple[Primitive | Composite | Self, int, int]:
        """
        :return: the value & the keyed_endorsements along the way
        """
        path = path.removeprefix(Node.PATH_SEPARATOR).removesuffix(Node.PATH_SEPARATOR)
        current_node = self
        components_left = current_node.rel_path(path).split(Node.PATH_SEPARATOR)
        if components_left[0] not in current_node.keyed_data:
            raise NodePathDoesNotExist(current_node, components_left)

        current_endorsements = current_node.keyed_endorsements[components_left[0]]
        total_endorsements = current_node.keyed_endorsements[components_left[0]]
        while components_left:
            current_node = current_node.keyed_data[components_left[0]]
            # if we have to continue from this point forward, get the node (if it exists)
            if len(components_left) > 1:
                eligible_node_list = list(filter(lambda i: isinstance(i, Node), current_node))
                if not eligible_node_list:
                    raise NodePathDoesNotExist(current_node, components_left[1:])

                assert len(eligible_node_list) == 1
                current_node = eligible_node_list[0]
            if not isinstance(current_node, Node):
                return current_node, current_endorsements, total_endorsements

            components_left = current_node.rel_path(path).split(Node.PATH_SEPARATOR)
            if components_left[0] not in current_node.keyed_data:
                raise NodePathDoesNotExist(current_node, components_left)

            current_endorsements += current_node.keyed_endorsements[components_left[0]]
            total_endorsements += current_node.total_endorsements

        return current_node, current_endorsements, total_endorsements

    def schema(self,
               inclusion_tolerance: float,
               required_tolerance: float,
               do_property_description_prompt: bool,
               title: str | None = None,
               description: str | None = None) -> dict:
        return self._aggragate_field_types(
            inclusion_tolerance,
            required_tolerance,
            do_property_description_prompt,
            title,
            description,
        )

    def telemetry(self) -> list[tuple[str, int, int, list[Primitive | Composite]]]:
        """
        To get % of times seen, use round((keyed_endorsements/total_endorsements)*100, 0)
        :return: A path-sorted list of [path.to.field, keyed_endorsements, total_endorsements, [types]]
        """
        data = []
        for k, v in self.keyed_data.items():
            keyed_endorsement = self.keyed_endorsements[k]

            data.append((
                self.abs_path(k),
                keyed_endorsement,
                self.total_endorsements,
                v
            ))

            # check for special case: nested endorsements
            nodes_in_values = list(filter(lambda i: isinstance(i, Node), v))
            # data sanity check: this list must never have more than 1 element
            assert len(nodes_in_values) <= 1
            if bool(len(nodes_in_values)):
                nested_telemetry = nodes_in_values[0].telemetry()
                for nested_record in nested_telemetry:
                    data.append((
                        nested_record[0],
                        keyed_endorsement + nested_record[1],
                        self.total_endorsements + nested_record[2],
                        nested_record[3]
                    ))
        return list(sorted(data, key=lambda item: item[0]))


def translate_to_primitive_schema_type(obj: Any | type | None) -> str:
    if obj is None:
        return "null"
    if isinstance(obj, str):
        return "string"
    if isinstance(obj, bool):
        return "boolean"
    if isinstance(obj, int):
        return "integer"
    if isinstance(obj, float):
        return "number"
    if isinstance(obj, (dict, Node)):
        return "object"
    if isinstance(obj, list):
        return "array"
    raise RuntimeError(obj)


# TODO rewrite this delusional piece of shit
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


class JsonConstraints(enum.Enum):
    @classmethod
    def from_string(cls, s: str) -> Self:
        raise NotImplementedError()


@dataclass
class UserOptions:
    # --from-jdump <filename>
    # --from-jarray <filename>
    input_file: Path
    input_is_jobj_dump_or_array: bool
    # --title <title str>
    title: str
    # --description <description str>
    description: str
    # --tolerance <tolerance> 0 < tolerance <= 1
    #   the higher this number is, the more things that will be included
    inclusion_tolerance: float
    # --required <tolerance>
    #   the higher this number is, the more things that will be included
    #   can be negative (no required)
    required_tolerance: float
    # --prompt-for-description
    do_property_description_prompt: bool
    # TODO enable when work is starting on constraints
    # --constraints type1,type2,...
    #   OPTIONAL: set to empty list by default
    # --output
    #   OPTIONAL
    #   use this argument with - for stdout to echo to stdout
    #   use this argument with /dev/null to silence completely (wtf)
    output: Path | None


def _parse_args(args) -> UserOptions:
    parser = argparse.ArgumentParser(description="json-roulette: a barebones json generator, for testing")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--from-jdump", type=str, default=None)
    group.add_argument("--from-jarray", type=str, default=None)
    parser.add_argument("--title", type=str, default="JSON Dump", required=False)
    parser.add_argument("--description", type=str, default="JSON Schema from a dump.", required=False)
    parser.add_argument("--tolerance", type=float, default=1.0, required=False)
    parser.add_argument("--required-tolerance", type=float, default=0.0, required=False)
    parser.add_argument("--prompt-for-description", default=False, action="store_true", required=False)
    parser.add_argument("--output", type=str, default=None, required=False)
    options = parser.parse_args(args)
    input_file = Path(options.from_jdump).expanduser() if options.from_jdump else None
    input_file = Path(options.from_jarray).expanduser() if options.from_jarray else input_file
    assert 0 < options.tolerance <= 1.0
    assert 0 <= options.required_tolerance <= 1.0
    return UserOptions(
        input_file=input_file,
        input_is_jobj_dump_or_array=bool(options.from_jdump),
        title=options.title,
        description=options.description,
        inclusion_tolerance=options.tolerance,
        required_tolerance=options.required_tolerance,
        do_property_description_prompt=options.prompt_for_description,
        output=options.output
    )


JOBJECT_SUPERSEDES_PRIMITIVES = True

OUTPUT_FILE = sys.stdout
if __name__ == "__main__":
    options = _parse_args(sys.argv[1:])
    try:
        OUTPUT_FILE = open(options.output, "w") if options.output else sys.stdout

        # root model
        model: Node = Node()
        with open(options.input_file, "r") as fj_dump:
            while jobj := read_jobj_incrementally(fj_dump):
                model.endorse_jobj(json.loads(jobj))

        print(model, file=OUTPUT_FILE)
        for r in model.telemetry():
            path, field_endorsements, total_endorsements, _ = r
            print(
                f"{path} was seen {round((field_endorsements / total_endorsements) * 100, 1)}% of the time.",
                file=OUTPUT_FILE
            )

        print(
            json.dumps(
                model.schema(
                    inclusion_tolerance=options.inclusion_tolerance,
                    required_tolerance=options.required_tolerance,
                    do_property_description_prompt=options.do_property_description_prompt,
                    title=options.title,
                    description=options.description
                ),
                indent=4
            ), file=OUTPUT_FILE
        )
        print(model.get_property(
            "$.agamist.chloralum.chloralum.plumery.dunged.inaccordant.torus.torus.agamist.gerant.eccoriate.torus.marshbanker.alisphenoidal.plumery"))
    finally:
        OUTPUT_FILE.close()
