#!/usr/bin/env python3.12
import argparse
import enum
import functools
import json
import sys
import types
from copy import copy
from dataclasses import dataclass
from itertools import zip_longest
from os import PathLike
from pathlib import Path
from typing import TextIO, Self, Iterator


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

type Node = object
type Key = str
type JsonPath = str
type JsonArray = list
type JsonObject = dict
type Composite = JsonObject | JsonArray
type Primitive = str | int | float | bool | None


class JPathDoesNotExist(Exception):
    pass


class JTypeDoesNotExist(Exception):
    pass


class JType(enum.Enum):
    NULL = types.NoneType
    STRING = str
    BOOLEAN = bool
    INTEGER = int
    NUMBER = float
    OBJECT = dict
    ARRAY = list

    def is_jobj(self):
        return self == JType.OBJECT

    def is_jarr(self):
        return self == JType.ARRAY

    def is_composite(self):
        return self.is_jobj() or self.is_jarr()

    def is_primitive(self):
        return not self.is_composite()

    def __str__(self):
        if self.is_jobj():
            return "object"
        if self.is_jarr():
            return "array"

        # primitive conversions here
        match self:
            case JType.NULL:
                return "null"
            case JType.STRING:
                return "string"
            case JType.BOOLEAN:
                return "boolean"
            case JType.INTEGER:
                return "integer"
            case JType.NUMBER:
                return "number"

        raise JTypeDoesNotExist(self.value)

    # use this function because _missing_ and __new__ are two pieces of shit
    @staticmethod
    def _new(_, value):
        if isinstance(value, types.NoneType):
            return JType.NULL
        if isinstance(value, (dict, Node)):
            return JType.OBJECT
        if isinstance(value, list):
            return JType.ARRAY
        if isinstance(value, str):
            return JType.STRING
        if isinstance(value, bool):
            return JType.BOOLEAN
        if isinstance(value, int):
            return JType.INTEGER
        if isinstance(value, float):
            return JType.NUMBER

        raise JTypeDoesNotExist(value, type(value))


JType.__new__ = JType._new

type JPathLike = PathLike | list[str] | str


class JPath:
    ROOT_NOTATION = "$"
    PATH_SEPARATOR = "."
    ARRAY_WILDCARD_NOTATION = "[]"

    def __init__(self, path: JPathLike | Self = None):
        if path is None:
            path = [JPath.ROOT_NOTATION]
        if isinstance(path, str):
            path = path.split(JPath.PATH_SEPARATOR)
        if isinstance(path, JPath):
            path = path.components

        # this must be true or we have massively fucked up
        path: list[str]
        assert isinstance(path, list)
        self.__components = path

    def __str__(self) -> str:
        return self.path

    def __eq__(self, other: JPathLike | Self) -> bool:
        if other is None:
            return False
        if not isinstance(other, (PathLike | str | JPath)):
            return False
        if not isinstance(other, JPath):
            other = JPath(other)
        for sp, op in zip_longest(self.components, other.components):
            if sp != op:
                return False

        return True

    def __truediv__(self, other: Self | str | int) -> Self:
        """
        :return: A new JPath
        """
        if not isinstance(other, JPath):
            return JPath(path=[*self.components, other])
        return JPath(path=self.components + other.components)

    @property
    def components(self) -> PathLike | list[str]:
        return copy(self.__components)

    @property
    def basename(self) -> str:
        if not self.components:
            raise JPathDoesNotExist()
        return self.components[-1]

    # TODO this outputs a non-standard way for JsonPath indices: for example, this will currently output $.thing.[0].thing2
    @property
    def path(self) -> str:
        # delegate to map
        def convert_array_indices(element) -> str:
            if isinstance(element, int):
                return f"[{element}]"
            return element
        return JPath.PATH_SEPARATOR.join(list(map(convert_array_indices, self.components)))

    @property
    def absolute(self):
        components = self.components
        if components[0] != JPath.ROOT_NOTATION:
            components.insert(0, JPath.ROOT_NOTATION)
        return JPath.PATH_SEPARATOR.join(components)

    @property
    def depth(self) -> int:
        return len(self.components)


@utils.auto_str
class JAggregate:
    def __init__(self, *values: ...):
        self.__values: list[...] = []
        self.__types: list[JType] = []
        self.__self_aggregations: int = 0
        self.__aggregations_per_type: dict[JType, int] = {}
        if values:
            self.aggragate(values)

    def __add__(self, other: Self) -> Self:
        if not isinstance(other, JAggregate):
            raise TypeError(f"Cannot aggregate different {other} type {type(other)}!")

        # sum the two statistic fields
        aggregation_sum = {}
        for k in set(self.__aggregations_per_type) | set(other.__aggregations_per_type):
            aggregation_sum[k] = self.__aggregations_per_type.get(k, 0) + other.__aggregations_per_type.get(k, 0)

        new_aggregate = JAggregate()
        new_aggregate.__values = list(set(self.values + other.values))
        new_aggregate.__types = list(set(self.types + other.types))
        new_aggregate.__aggregations_per_type = aggregation_sum
        new_aggregate.__self_aggregations = self.__self_aggregations + other.__self_aggregations
        return new_aggregate

    def __aggregate_jtype(self, jtype: JType):
        if jtype not in self.__types:
            self.__types.append(jtype)

        if jtype not in self.__aggregations_per_type:
            self.__aggregations_per_type[jtype] = 1
            return
        self.__aggregations_per_type[jtype] += 1

    @property
    def values(self):
        return copy(self.__values)

    @property
    def types(self) -> list[JType]:
        return copy(self.__types)

    @property
    def aggregations(self) -> int:
        """
        :return: the total number of aggregations of any value (self)
        """
        return self.__self_aggregations

    def frequency(self, jtype: JType) -> float:
        """
        Get appearance statistics for a specific jtype.
        :return: % of appearances
        """
        return self.__aggregations_per_type[jtype] / functools.reduce(lambda x, y: x + y, list(self.__aggregations_per_type.values()), 0)

    def aggragate(self, *values):
        """
        Aggregate N valid JType candidates to this container.
        """
        if not values:
            return
        self.__self_aggregations += 1
        for v in values:
            self.__values.append(v)
            self.__aggregate_jtype(JType(v))


# TODO rename to JAggregator
# TODO provide ways to access all data sanely w/ BFS (walk)
#      - BFS should probably return the JPath & the data relevant to the field we're currently walking through
# TODO support arrays before continuing development! this might backfire
# TODO move all schema & output-relevant logic from here to JSchema
# this class must serve the sole & explicit role of collecting the data correctly; this MUST NOT
#  have any logic regarding types etc; just collect the data sanely (!)
@utils.auto_str
class Node:
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
        return self.path.split(JPath.PATH_SEPARATOR)[-1]

    def abs_path(self, k: Key) -> JsonPath:
        """
        :return: the absolute path to a key
        """
        return JPath.PATH_SEPARATOR.join([*self.path.split(JPath.PATH_SEPARATOR), k])

    def rel_path(self, path: Key) -> JsonPath:
        """
        :return: the relative path to a key, from the current node
        """
        return path.removeprefix(self.path).removeprefix(JPath.PATH_SEPARATOR)

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
        schema = {"type": str(JType(self))}
        property_types = {}
        for key, values in self.keyed_data.items():
            property_types[key] = []
            values: list
            for item in values:
                property_types[key].append(str(JType(item)))

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
        path = path.removeprefix(JPath.PATH_SEPARATOR).removesuffix(JPath.PATH_SEPARATOR)
        current_node = self
        components_left = current_node.rel_path(path).split(JPath.PATH_SEPARATOR)
        if components_left[0] not in current_node.keyed_data:
            raise JPathDoesNotExist(current_node, components_left)

        current_endorsements = current_node.keyed_endorsements[components_left[0]]
        total_endorsements = current_node.keyed_endorsements[components_left[0]]
        while components_left:
            current_node = current_node.keyed_data[components_left[0]]
            # if we have to continue from this point forward, get the node (if it exists)
            if len(components_left) > 1:
                eligible_node_list = list(filter(lambda i: isinstance(i, Node), current_node))
                if not eligible_node_list:
                    raise JPathDoesNotExist(current_node, components_left[1:])

                assert len(eligible_node_list) == 1
                current_node = eligible_node_list[0]
            if not isinstance(current_node, Node):
                return current_node, current_endorsements, total_endorsements

            components_left = current_node.rel_path(path).split(JPath.PATH_SEPARATOR)
            if components_left[0] not in current_node.keyed_data:
                raise JPathDoesNotExist(current_node, components_left)

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


def walk(jelement: JType.OBJECT.value | JType.ARRAY.value, root_path: JPath | str = None) -> Iterator[tuple[JPath, ...]]:
    def is_delegate(el: JType.OBJECT.value | JType.ARRAY.value) -> bool:
        return JType(el).is_composite()

    jtype = JType(jelement)
    jpath = JPath(root_path)
    if jtype.is_jobj():
        for k, v in jelement.items():
            if is_delegate(v):
                yield from walk(v, root_path=jpath / k)
                continue
            yield jpath / k, v
        return

    if jtype.is_jarr():
        for i in range(len(jelement)):
            v = jelement[i]
            if is_delegate(v):
                yield from walk(v, root_path=jpath / i)
                continue
            yield jpath / i, v
        return

    raise JTypeDoesNotExist(f"Invalid root jelement {jelement} type {jtype}")


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


def main(options) -> None:
    global OUTPUT_FILE
    try:
        OUTPUT_FILE = open(options.output, "w") if options.output else sys.stdout

        with open(options.input_file, "r") as fj_dump:
            while jobj := read_jobj_incrementally(fj_dump):
                for tup in walk(json.loads(jobj)):
                    jpath, *vals = tup
                    print(jpath, vals)

        # # root model
        # model: Node = Node()
        # with open(options.input_file, "r") as fj_dump:
        #     while jobj := read_jobj_incrementally(fj_dump):
        #         model.endorse_jobj(json.loads(jobj))
        #
        # print(model, file=OUTPUT_FILE)
        # for r in model.telemetry():
        #     path, field_endorsements, total_endorsements, _ = r
        #     print(
        #         f"{path} was seen {round((field_endorsements / total_endorsements) * 100, 1)}% of the time.",
        #         file=OUTPUT_FILE
        #     )
        #
        # print(
        #     json.dumps(
        #         model.schema(
        #             inclusion_tolerance=options.inclusion_tolerance,
        #             required_tolerance=options.required_tolerance,
        #             do_property_description_prompt=options.do_property_description_prompt,
        #             title=options.title,
        #             description=options.description
        #         ),
        #         indent=4
        #     ), file=OUTPUT_FILE
        # )
        # # noinspection SpellCheckingInspection
        # print(model.get_property(
        #     "$.agamist.chloralum.chloralum.plumery.dunged.inaccordant.torus.torus.agamist.gerant.eccoriate.torus.marshbanker.alisphenoidal.plumery"))
    finally:
        OUTPUT_FILE.close()


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
    main(_parse_args(sys.argv[1:]))
