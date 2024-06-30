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
from typing import TextIO, Iterator, TypeAlias
import platform

Self = object
if int(platform.python_version_tuple()[1]) >= 11:
    from typing import Self


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
    "4110569f2ad92677cdc94002d3c52c9440de4f434636de580140e52b6f1d1d3b"
)


class JPathDoesNotExist(Exception):
    pass


class JTypeDoesNotExist(Exception):
    pass


JTypeObjectCandidate: TypeAlias = dict
JTypeArrayCandidate: TypeAlias = list
JTypeCompositeCandidate: TypeAlias = JTypeObjectCandidate | JTypeArrayCandidate
JTypePrimitiveCandidate: TypeAlias = types.NoneType | str | bool | int | float


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
        if isinstance(value, dict):
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

JPathLike = PathLike | list[str] | str


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

    # map delegates
    #  __convert_array_indices
    #  __convert_string_indices
    # noinspection PyMethodMayBeStatic
    def __convert_array_indices(self, element) -> str:
        if isinstance(element, int):
            return f"[{element}]"
        return element

    # noinspection PyMethodMayBeStatic
    def __convert_string_indices(self, element) -> int:
        raise NotImplementedError()  # TODO

    def __str__(self) -> str:
        return self.path

    def __repr__(self) -> str:
        return self.path

    def __hash__(self) -> int:
        return hash(self.path)

    def __contains__(self, other: Self) -> bool:
        assert isinstance(other, JPath)
        for sp, op in zip_longest(self.components, other.components):
            if sp != op:
                return False
        return True

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
    def parent(self) -> Self | None:
        # check for tld side-case
        if self.rank == 1:
            return None
        return JPath(self.components[:-1])

    @property
    def components(self) -> PathLike | list[str]:
        return copy(self.__components)

    @property
    def basename(self) -> str:
        if not self.components:
            raise JPathDoesNotExist()
        return self.components[-1]

    @property
    def path(self) -> str:
        return JPath.PATH_SEPARATOR.join(list(map(self.__convert_array_indices, self.components)))

    @property
    def absolute(self):
        components = self.components
        if components[0] != JPath.ROOT_NOTATION:
            components.insert(0, JPath.ROOT_NOTATION)
        return JPath.PATH_SEPARATOR.join(components)

    @property
    def rank(self) -> int:
        return len(self.components)

    def is_jarr(self) -> bool:
        last_component = self.components[-1]
        return last_component == JPath.ARRAY_WILDCARD_NOTATION or JType(last_component) == JType.INTEGER

    @classmethod
    def root(cls):
        return cls(JPath.ROOT_NOTATION)

    @classmethod
    def bfs_sort(cls, jpaths: set[str]) -> list[Self]:
        """
        Natural sort a set of keys, & return the normalized set of paths.
        The expected outcome should look like this ('bfs' sort):
        [
            $,
            $.path,
            $.path2,
            $.path.inner,
            $.path.[].inner,
            $.path2.inner.inner
        ]
        """
        return sorted(set(map(lambda p: JPath(p), jpaths)), key=lambda i: i.rank)


@utils.auto_str
class JAggregate:
    def __init__(self, value: ... = None):
        self.__types: list[JType] = []
        self.__self_aggregations: int = 0
        self.__aggregations_per_type: dict[JType, int] = {}
        if value:
            self.aggregate(value)

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
        return self.__aggregations_per_type[jtype] / functools.reduce(lambda x, y: x + y,
                                                                      list(self.__aggregations_per_type.values()), 0)

    def aggregate(self, value: ...):
        """
        Aggregate a valid JType candidate to this container.
        :param value: value to aggregate
        """
        jtype = JType(value)
        self.__self_aggregations += 1

        # self.__values.append(v)
        if jtype not in self.__types:
            self.__types.append(jtype)

        if jtype not in self.__aggregations_per_type:
            self.__aggregations_per_type[jtype] = 1
            return
        self.__aggregations_per_type[jtype] += 1


# use this because __repr__ is a piece of shit
JAggregate.__repr__ = JAggregate.__str__


# this class must serve the sole & explicit role of collecting the data correctly; this MUST NOT
#  have any logic regarding types etc.; just collect the data sanely (!)
# The reason to build the schema in a bottom-up way is, so we can keep track of our father's type
#  which is necessary, in order to know 'where' to place our jschema
#  obviously, this 'data' is in the json path itself:
#  - when our parent is a regular string, it means we're part of an object;
#  - when our parent is an integer, it means we're part of an array
#  check this information with .parent().is_jarr
@utils.auto_str
class JAggregator:
    def __init__(self):
        # dict w/ NORMALIZED keys!
        self.aggregates: dict[str, JAggregate] = {}

    def aggregate(self, jpath: JPath, value):
        assert isinstance(jpath, JPath)
        if jpath.path not in self.aggregates:
            self.aggregates[jpath.path] = JAggregate()

        self.aggregates[jpath.path].aggregate(value)

    def collapse_arrays(self):
        """
        this method was created because of the following terrible bug:
        {
            "test": [
                1,
                2,
                3
            ]
        }
        {
            "test": [
                "string"
            ]
        }
        when this is parsed, it'll output that $.test.[] hz is 2.0, which it's obviously not
        this happens because for every $.test.[0], $.test.[1], $.test.[2], $.test.[3]
        the path is normalized, which means that
        $.test.[] is added 4 times

        Another example:
        $.explode.foreboding.abash.tenpins.enchanting.[] hz: 6.0
        ... because of
        $.explode.foreboding.abash.tenpins.enchanting.[0]
        $.explode.foreboding.abash.tenpins.enchanting.[1]
        $.explode.foreboding.abash.tenpins.enchanting.[2]
        $.explode.foreboding.abash.tenpins.enchanting.[3]
        $.explode.foreboding.abash.tenpins.enchanting.[4]
        $.explode.foreboding.abash.tenpins.enchanting.[5]
        """
        raise NotImplementedError()  # TODO implement

    def get(self, key: JPath) -> JAggregate | None:
        assert isinstance(key, JPath)
        if key.path not in self.aggregates:
            return None

        return self.aggregates[key.path]

    def reverse_treeline_iterator(self) -> Iterator[tuple[JPath, JAggregate]]:
        """
        Return a reverse treeline of all given nodes in a path.
        Will always return ROOT as the final result.
        For an example output, this will return the following outputs for the top level output:
        $.agamist.chloralum.chloralum.plumery.dunged.inaccordant.torus.torus.agamist.gerant.eccoriate.torus.marshbanker.alisphenoidal.anthropidae
        $.agamist.chloralum.chloralum.plumery.dunged.inaccordant.torus.torus.agamist.gerant.eccoriate.torus.marshbanker.alisphenoidal
        $.agamist.chloralum.chloralum.plumery.dunged.inaccordant.torus.torus.agamist.gerant.eccoriate.torus.marshbanker
        $.agamist.chloralum.chloralum.plumery.dunged.inaccordant.torus.torus.agamist.gerant.eccoriate.torus
        $.agamist.chloralum.chloralum.plumery.dunged.inaccordant.torus.torus.agamist.gerant.eccoriate
        $.agamist.chloralum.chloralum.plumery.dunged.inaccordant.torus.torus.agamist.gerant
        $.agamist.chloralum.chloralum.plumery.dunged.inaccordant.torus.torus.agamist
        $.agamist.chloralum.chloralum.plumery.dunged.inaccordant.torus.torus
        $.agamist.chloralum.chloralum.plumery.dunged.inaccordant.torus
        $.agamist.chloralum.chloralum.plumery.dunged.inaccordant
        $.agamist.chloralum.chloralum.plumery.dunged
        $.agamist.chloralum.chloralum.plumery
        $.agamist.chloralum.chloralum
        $.agamist.chloralum
        $.agamist
        $
        ... starting from the beginning (on another node)
        """

        def yield_ancestry(jpath: JPath) -> Iterator[JPath]:
            parent = jpath.parent
            yield parent
            # don't return null if no parents (exit condition)
            if not parent.parent:
                return
            yield from yield_ancestry(parent)

        index = 0
        jpaths = JPath.bfs_sort(set(self.aggregates.keys()))[::-1]
        while index < len(jpaths):
            jp = jpaths[index]
            yield jp, self.aggregates[jp.path]

            jpaths.remove(jp)
            # don't return null if no parents
            if not jp.parent:
                continue

            ancestry = yield_ancestry(jp)
            for ancestor in ancestry:
                if ancestor in jpaths:
                    jpaths.remove(ancestor)
                yield ancestor, self.aggregates[ancestor.path]

        return


# This class is responsible for the recursive construction of the $ json-schema. This means the following:
# - TODO finish responsibilities, we need to be verbose here
class JSchema:
    def __init__(self, jaggregator: JAggregator):
        # TODO: add UserOptions
        self.model = jaggregator

    def schema(self) -> dict:
        """
        :return: a valid json-schema
        """
        root_aggregate = self.model.get(JPath.root())
        for jpath, jaggregate in self.model.reverse_treeline_iterator():
            # get the parent if it exists, otherwise get the root, which can only compare to itself
            parent = self.model.get(jpath.parent) if jpath.parent else root_aggregate

            print(f"{jpath} hz: {jaggregate.aggregations / parent.aggregations}")
            jaggregate_types: list[JType] = jaggregate.types
            for jt in jaggregate_types:
                print(f"{jt} hz: {jaggregate.frequency(jt)}")

        raise NotImplementedError()  # TODO implement


def walk(jelement: JTypeCompositeCandidate, root_path: JPath | str = None) -> Iterator[tuple[...]]:
    """
    :raises JTypeDoesNotExist: if $ is not a jcomposite
    :returns: JPath, object
    """

    def delegate_obj(jl: JTypeObjectCandidate, jp: JPath):
        for k, v in jl.items():
            # sanity check: json allows for keys named "my.thing.etc" but this will massively fuck up everything
            assert JPath.PATH_SEPARATOR not in k
            if JType(v).is_composite():
                yield from walk(v, root_path=jp / k)
                continue
            yield jp / k, v
        return

    def delegate_jarr(jl: JTypeArrayCandidate, jp: JPath):
        for i in range(len(jl)):
            v = jl[i]
            # sanity check is not needed here because we yield via the array wildcard
            if JType(v).is_composite():
                yield from walk(v, root_path=jp / i)
                continue
            yield jp / i, v
        return

    delegates = {
        JType.OBJECT: delegate_obj,
        JType.ARRAY: delegate_jarr
    }

    jtype = JType(jelement)
    jpath = JPath(root_path)
    if not jtype.is_composite():
        raise JTypeDoesNotExist(f"Invalid root jelement {jelement} type {jtype}")

    yield jpath, jelement
    yield from delegates[jtype](jelement, jpath)
    return


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

        # root model
        model: JAggregator = JAggregator()
        with open(options.input_file, "r") as fj_dump:
            while jobj := read_jobj_incrementally(fj_dump):
                for jpath, value in walk(json.loads(jobj)):
                    model.aggregate(jpath, value)

        # if we're dealing with unordered arrays, collapse the contents to avoid
        #  index-significant handling
        if options.unordered_arrays:
            model.collapse_arrays()

        print(json.dumps(JSchema(model).schema(), indent=4))

    finally:
        OUTPUT_FILE.close()


@dataclass
class UserOptions:
    input_file: Path
    # --title <title str>
    title: str
    # --description <description str>
    description: str
    # --prompt-for-description
    do_property_description_prompt: bool
    # --ordered
    #   if the inverse of the below is set, the indexes within arrays will be considered important
    unordered_arrays: bool
    # TODO reenable when we figure out what exactly this means business-wise
    # --tolerance <tolerance> 0 < tolerance <= 1
    #   the higher this number is, the more things that will be included
    # inclusion_tolerance: float
    # --required <tolerance>
    #   the higher this number is, the more things that will be included
    #   can be negative (no required)
    required_tolerance: float
    # TODO enable when work is starting on constraints
    # --constraints type1,type2,...
    #   OPTIONAL: set to empty list by default
    # --output
    #   OPTIONAL
    #   use this argument with - for stdout to echo to stdout
    #   use this argument with /dev/null to silence completely (wtf)
    output: Path | None


def _parse_args(args) -> UserOptions:
    parser = argparse.ArgumentParser(description="jproof: a json-schema generator")
    parser.add_argument("--from-jdump", type=str, default=None, required=True)
    parser.add_argument("--title", type=str, default="JSON Dump", required=False)
    parser.add_argument("--description", type=str, default="JSON Schema from a dump.", required=False)
    parser.add_argument("--prompt-for-description", default=False, action="store_true", required=False)
    parser.add_argument("--ordered", default=False, action="store_true", required=False)
    parser.add_argument("--tolerance", type=float, default=1.0, required=False)
    parser.add_argument("--required-tolerance", type=float, default=0.0, required=False)
    parser.add_argument("--output", type=str, default=None, required=False)
    options = parser.parse_args(args)
    input_file = Path(options.from_jdump).expanduser()
    assert 0 < options.tolerance <= 1.0
    assert 0 <= options.required_tolerance <= 1.0
    return UserOptions(
        input_file=input_file,
        title=options.title,
        description=options.description,
        do_property_description_prompt=options.prompt_for_description,
        unordered_arrays=not options.ordered,
        # inclusion_tolerance=options.tolerance,
        required_tolerance=options.required_tolerance,
        output=options.output
    )


OUTPUT_FILE = sys.stdout
if __name__ == "__main__":
    main(_parse_args(sys.argv[1:]))
