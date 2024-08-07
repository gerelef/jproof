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


def auto_str(cls):
    """Automatically implements __str__ for any class."""

    def __str__(self):
        return '%s(%s)' % (
            type(self).__name__,
            ', '.join('%s=%s' % item for item in vars(self).items())
        )

    cls.__str__ = __str__
    return cls


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
        for sp, op in zip(self.components, other.components):
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

    def rebase(self, new_base: Self) -> Self:
        """
        Rebase the current path to a new root. For example, take the following two jpaths:
        $.basifier                             <- new base
        $.basifier.huzzaed.[].sedimentary      <- "self"

        with components:
        $ basifier
        $ basifier huzzaed [] sedimentary

        ------- self, after: -------

        $ huzzaed [] sedimentary
        :returns: a rebased new JPath
        """
        assert isinstance(new_base, JPath)
        assert self in new_base

        new_components = new_base.components
        self_components = self.components
        while len(new_components) > 0 and ((nc := new_components.pop(0)) == (sc := self_components.pop(0))):
            pass

        self_components.insert(0, JPath.ROOT_NOTATION)
        return JPath(self_components)

    def is_root(self) -> bool:
        return len(self.components) == 1 and self.components[0] == JPath.ROOT_NOTATION

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


@auto_str
class JAggregate:
    def __init__(self, value: ... = None):
        self.__types: list[JType] = []
        self.__self_aggregations: int = 0
        self.__aggregations_per_type: dict[JType, int] = {}
        if value:
            self.aggregate(value)

    def __add__(self, other: Self) -> Self:
        assert isinstance(other, JAggregate)
        types: list[JType] = list(set(self.__types + other.__types))
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

    def aggregations(self, jtype: JType = None) -> int:
        """
        :return: the total number of aggregations of any value (self)
        """
        if jtype:
            assert jtype in self.__aggregations_per_type
            return self.__aggregations_per_type[jtype]
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
@auto_str
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

        # step 2: soft-cap ARRAY aggregate to the parent's ARRAY appearances,
        #  as arrays are unique multi-value containers that must be counted
        #  ... as explained (in detail) in this method's docs
        sorted_keys = JPath.nsorted(set(self.__aggregates.keys()))
        for jpath in sorted_keys:
            if not jpath.is_jarr():
                continue
            # ... if there IS a parent aggregate of the array
            if parent_aggregate := self.get(jpath.parent):
                self.get(jpath).cap(parent_aggregate.aggregations(jtype=JType.ARRAY))

    def get(self, key: JPath | None, or_else: object = None) -> JAggregate | None:
        """
        :return: Aggregate that matches the key, otherwise return or_else (by default, None).
        """
        assert key is None or isinstance(key, JPath)
        ret_val = self.__aggregates.get(key)
        return ret_val if ret_val is not None else or_else

    def treeline_iterator(self) -> Iterator[tuple[JPath, JAggregate]]:
        stack = []
        jp: JPath
        ja: JAggregate
        for jp, ja in self.reverse_treeline_iterator():
            stack.insert(0, (jp, ja))
            if jp.is_root():
                yield from stack
                stack.clear()

        return

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
# TODO finish responsibilities, we need to be verbose here
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

        # step 1:
        #  iterate, from the bottom to the top (reverse) any treeline
        #  on every step, produce a schema to be assigned to its parent
        for jpath, jaggregate in self.model.reverse_treeline_iterator():
            # get the parent if it exists, otherwise get the root, which can only compare to itself
            parent = self.model.get(jpath.parent, or_else=root_aggregate)
            print(jpath)
            # print(f"{jpath} hz: {jaggregate.aggregations() / parent.aggregations()}")
            # for jt in jaggregate.types:
            #     print(f"{jpath}:{jt} hz: {jaggregate.frequency(jt)}")

        # step n:
        #  assign metadata to the root json object.
        # TODO

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


def read_jroot_incrementally(f: TextIO) -> dict | list | None:
    """
    :return: a string representation of a json object from file
    """

    def is_start_token(c: str) -> bool:
        return c == "[" or c == "{"

    def is_end_token(sc: str, ec: str) -> bool:
        if sc == "{" and ec == "}":
            return True
        if sc == "[" and ec == "]":
            return True
        return False

    start_prev = 0
    while True:
        start_c = f.read(1)
        start = f.tell()
        if start_prev == start:
            return None
        start_prev = start
        if is_start_token(start_c):
            start -= 1
            break

    end_prev = start
    while True:
        end_c = f.read(1)
        end = f.tell()
        if end_prev == end:
            raise json.JSONDecodeError("No ending character!", f.read(end - start), end)
        end_prev = end
        if is_end_token(start_c, end_c):
            try:
                f.seek(start)
                return json.loads(f.read(end - start))
            except json.JSONDecodeError:
                f.seek(end)


def main(options) -> None:
    output_file = None
    try:
        output_file = open(options.output, "w") if options.output else sys.stdout
        JPath.PATH_SEPARATOR = options.sep  # set user-defined json separator for insane case where keys have a dot

        # root model
        model: JAggregator = JAggregator()
        with open(options.input_file, "r") as fj_dump:
            while jroot := read_jroot_incrementally(fj_dump):
                for jpath, value in walk(jroot):
                    model.aggregate(jpath, value)

        # if we're dealing with unordered arrays, normalize the contents to avoid
        #  index-significant handling
        if options.unordered_arrays:
            model.normalize()

        print(json.dumps(JSchema(model).schema(), indent=4 if options.pretty else None), file=output_file)
    finally:
        if output_file:
            output_file.close()


@dataclass
class UserOptions:
    input_file: Path
    # --title <title str>
    title: str
    # --description <description str>
    description: str
    # --prompt-for-description
    # do_property_description_prompt: bool
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
    # required_tolerance: float
    # TODO enable when work is starting on constraints
    # --constraints type1,type2,...
    #   OPTIONAL: set to empty list by default
    # --pretty
    #  OPTIONAL: will pretty print or not
    pretty: bool
    # --output
    #   OPTIONAL
    #   use this argument with - for stdout to echo to stdout
    #   use this argument with /dev/null to silence completely (wtf)
    output: Path | None
    # --separator, --sep
    sep: str


def _parse_args(args) -> UserOptions:
    parser = argparse.ArgumentParser(description="jproof: a json-schema generator")
    parser.add_argument("path", type=str)
    parser.add_argument("--title", type=str, default="JSON Dump", required=False)
    parser.add_argument("--description", type=str, default="JSON Schema from a dump.", required=False)
    # parser.add_argument("--prompt-for-description", default=False, action="store_true", required=False)
    parser.add_argument("--ordered", default=False, action="store_true", required=False)
    # parser.add_argument("--tolerance", type=float, default=1.0, required=False)
    # parser.add_argument("--required-tolerance", type=float, default=0.0, required=False)
    parser.add_argument("--pretty", default=False, action="store_true", required=False)
    parser.add_argument("--output", type=str, default=None, required=False)
    parser.add_argument("--separator", "--sep", type=str, default=None, required=False)
    options = parser.parse_args(args)
    input_file = Path(options.path).expanduser()
    # TODO implement verbose value range check from arguments, sys.exit 2 if wrong
    return UserOptions(
        input_file=input_file,
        title=options.title,
        description=options.description,
        # do_property_description_prompt=options.prompt_for_description,
        unordered_arrays=not options.ordered,
        # inclusion_tolerance=options.tolerance,
        # required_tolerance=options.required_tolerance,
        pretty=options.pretty,
        output=options.output,
        sep=options.separator if options.separator else JPath.PATH_SEPARATOR
    )


if __name__ == "__main__":
    main(_parse_args(sys.argv[1:]))
