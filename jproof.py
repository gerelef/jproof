#!/usr/bin/env python3.12
import argparse
import enum
import functools
import json
import platform
import sys
import types
from copy import copy
from dataclasses import dataclass
from itertools import zip_longest
from os import PathLike
from pathlib import Path
from typing import TextIO, Iterator, TypeAlias

Self = object
if int(platform.python_version_tuple()[1]) >= 11:
    from typing import Self

try:
    import readline
except ImportError:
    print(
        "FATAL: jproof REQUIRES the 'readline' module to be available. "
        "This is often provided from the GNU readline package. "
        "If you are unaware of what this means, consult the documentation and/or your system administrator.",
        file=sys.stderr
    )
    sys.exit(1)


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

    def is_jobj(self) -> bool:
        return self == JType.OBJECT

    def is_jarr(self) -> bool:
        return self == JType.ARRAY

    def is_composite(self) -> bool:
        return self.is_jobj() or self.is_jarr()

    def is_primitive(self) -> bool:
        return not self.is_composite()

    def __str__(self) -> str:
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
        if isinstance(path, str | list):
            path = self.__jp_from_path(path)
        if isinstance(path, JPath):
            path = path.components

        # this must be true or we have massively fucked up
        path: list[str]
        assert isinstance(path, list)
        self.__components: list[str | int] = path

    # map delegates
    #  __convert_array_indices
    #  __convert_string_indices
    # noinspection PyMethodMayBeStatic
    def __convert_int_to_array_indices(self, element: str | int) -> str:
        if isinstance(element, int):
            return f"[{element}]"
        return element

    # noinspection PyMethodMayBeStatic
    def __convert_array_to_int_indices(self, element: str) -> str | int:
        has_array_wrapping = isinstance(element, str) and element.startswith("[") and element.endswith("]")
        # at least 3 elements required to display [n], along w/ the array wrapping
        if has_array_wrapping and len(element) >= 3:
            return int(element[1:-1])
        return element

    def __jp_from_path(self, jpathlike: list[str] | str) -> list[str | int]:
        if isinstance(jpathlike, str):
            jpathlike = jpathlike.split(JPath.PATH_SEPARATOR)

        return list(map(self.__convert_array_to_int_indices, jpathlike))

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
        return str(self.components[-1])

    @property
    def path(self) -> str:
        return JPath.PATH_SEPARATOR.join(list(map(self.__convert_int_to_array_indices, self.components)))

    @property
    def absolute(self) -> str:
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

    def normalize(self) -> Self:
        """
        :return: A new object, which is the normalized version of self.
        """
        normalized_components = []
        for component in self.components:
            if isinstance(component, int):
                normalized_components.append(JPath.ARRAY_WILDCARD_NOTATION)
                continue
            normalized_components.append(component)

        return JPath(normalized_components)

    @classmethod
    def root(cls) -> Self:
        return cls(JPath.ROOT_NOTATION)

    @classmethod
    def nsorted(cls, jpaths: set[Self]) -> list[Self]:
        """
        Natural sort a set of keys.
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
        return sorted(jpaths, key=lambda i: i.rank)


@utils.auto_str
class JAggregate:
    def __init__(self, value: ... = None):
        self.__types: list[JType] = []
        self.__self_aggregations: int = 0
        self.__aggregations_per_type: dict[JType, int] = {}
        if value:
            self.aggregate(value)

    def __add__(self, other: Self) -> Self:
        assert isinstance(other, JAggregate)
        types: list[JType] = self.__types + other.__types
        self_aggregations: int = self.__self_aggregations + other.__self_aggregations
        aggregations_per_type: dict[JType, int] = copy(self.__aggregations_per_type)
        for other_aggregations_type, other_aggregations_appearances in other.__aggregations_per_type.items():
            if other_aggregations_type not in aggregations_per_type:
                aggregations_per_type[other_aggregations_type] = other_aggregations_appearances
                continue

            aggregations_per_type[other_aggregations_type] += other_aggregations_appearances

        new = JAggregate()
        new.__types = types
        new.__self_aggregations = self_aggregations
        new.__aggregations_per_type = aggregations_per_type
        return new

    @property
    def types(self) -> list[JType]:
        return copy(self.__types)

    @property
    def aggregations(self) -> int:
        """
        :return: the total number of aggregations of any value (self)
        """
        return self.__self_aggregations

    def cap(self, total_aggregations: int) -> None:
        """
        Soft-cap total aggregations to provided number.
        """
        if self.__self_aggregations > total_aggregations:
            self.__self_aggregations = total_aggregations

    def frequency(self, jtype: JType) -> float:
        """
        Get appearance statistics for a specific jtype.
        :return: % of appearances
        """
        return self.__aggregations_per_type[jtype] / functools.reduce(lambda x, y: x + y,
                                                                      list(self.__aggregations_per_type.values()), 0)

    def aggregate(self, value: ...) -> None:
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
        self.__aggregates: dict[JPath, JAggregate] = {}

    def aggregate(self, jpath: JPath, value) -> None:
        assert isinstance(jpath, JPath)
        if jpath.path not in self.__aggregates:
            self.__aggregates[jpath] = JAggregate()

        self.__aggregates[jpath].aggregate(value)

    def normalize(self) -> None:
        """
        collapse_arrays; this method was created because of the following terrible bug:
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
        ...
        The consequence of this function, will be the following:
        - the aggregates will be summed.
        - if at least one element existed, whatever it was, the frequency will be capped to it's parent
           for example, if `$.explode.foreboding.abash.tenpins.enchanting` was seen 2 times,
           the 'appearances' array will be capped to 2.
        """
        # step 1: normalize everything in the existing dict
        normalized_aggregates: dict[JPath, JAggregate] = {}
        jpaths: list[JPath] = list(self.__aggregates.keys())
        for jpath in jpaths:
            normalized_jp = jpath.normalize()
            jaggregate = self.__aggregates.pop(jpath)
            if normalized_jp not in normalized_aggregates:
                normalized_aggregates[normalized_jp] = jaggregate
                continue
            normalized_aggregates[normalized_jp] += jaggregate

        self.__aggregates = normalized_aggregates

        # step 2: soft-cap each aggregate to its parent aggregations, as it literally CANNOT have a value bigger than 1
        #  ... as explained (in detail) in this method's docs
        for jpath, aggregate in self.__aggregates.items():
            if parent_aggregate := self.get(jpath.parent):
                aggregate.cap(parent_aggregate.aggregations)

    def get(self, key: JPath | None, or_else: object = None) -> JAggregate | None:
        assert key is None or isinstance(key, JPath)
        if key is None or key.path not in self.__aggregates:
            return or_else

        return self.__aggregates[key]

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
        jpaths = JPath.nsorted(set(self.__aggregates.keys()))[::-1]
        while index < len(jpaths):
            jp = jpaths[index]
            yield jp, self.__aggregates[jp]

            jpaths.remove(jp)
            # don't return null if no parents
            if not jp.parent:
                continue

            ancestry = yield_ancestry(jp)
            for ancestor in ancestry:
                if ancestor in jpaths:
                    jpaths.remove(ancestor)
                yield ancestor, self.__aggregates[ancestor]

        return


# This class is responsible for the recursive construction of the $ json-schema. This means the following:
# - TODO finish responsibilities, we need to be verbose here
# TODO add UserOptions
class JSchema:
    def __init__(self, jaggregator: JAggregator):
        self.model = jaggregator

    def schema(self) -> dict:
        """
        :return: a valid json-schema
        """
        root_aggregate = self.model.get(JPath.root())
        assert root_aggregate is not None  # sanity check

        for jpath, jaggregate in self.model.reverse_treeline_iterator():
            # get the parent if it exists, otherwise get the root, which can only compare to itself
            parent = self.model.get(jpath.parent, or_else=root_aggregate)

            print(f"{jpath} hz: {jaggregate.aggregations / parent.aggregations}")
            # FIXME there's an unexpected behaviour here; if there is no content
            #  in the array, the do not get counted for the field, however the nested
            #  fields do indeed see it; check "test" jarr key behaviour for more insight
            # for jt in jaggregate.types:
            #     print(f"{jt} hz: {jaggregate.frequency(jt)}")

        raise NotImplementedError()  # TODO implement


def walk(jelement: JTypeCompositeCandidate, root_path: JPath | str = None) -> Iterator[tuple[...]]:
    """
    :raises JTypeDoesNotExist: if $ is not a jcomposite
    :returns: JPath, object
    """

    def delegate_obj(jl: JTypeObjectCandidate, jp: JPath) -> Iterator[tuple[...]]:
        for k, v in jl.items():
            # sanity check: json allows for keys named "my.thing.etc" but this will massively fuck up everything
            assert JPath.PATH_SEPARATOR not in k
            if JType(v).is_composite():
                yield from walk(v, root_path=jp / k)
                continue
            yield jp / k, v
        return

    def delegate_jarr(jl: JTypeArrayCandidate, jp: JPath) -> Iterator[tuple[...]]:
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

        # if we're dealing with unordered arrays, normalize the contents to avoid
        #  index-significant handling
        if options.unordered_arrays:
            model.normalize()

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
