#!/usr/bin/env python3.12
import json
from pprint import pprint
from sys import argv
from types import NoneType
from typing import TextIO, Any, Self

type Key = str
type JsonArray = list
type JsonObject = dict
type Composite = JsonObject | JsonArray | Node
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


@utils.auto_str
class Node:
    def __init__(self, path: list = None):
        self.path = path if path else []
        self.keyed_data: dict[Key, list[Primitive | Composite]] = {}
        self.keyed_endorsements: dict[Key, int] = {}
        self.total_endorsements = 0

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
        jobj = Node([*self.path, k])
        jobj.endorse_jobj(o)
        return jobj

    def endorse_jobj(self, jobj: JsonObject | dict) -> Self:
        self.total_endorsements += 1
        for k, v in jobj.items():
            self._increment_endorsement(k)
            if k not in self.keyed_data:
                self.keyed_data[k] = [self._create_node(k, v)]
                continue

            self.keyed_data[k].append(self._create_node(k, v))

        return self

    def _aggragate_field_types(self) -> dict[str, str | list[str] | dict]:
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
                field_has_own_properties = "object" in jtype if JOBJECT_SUPERSEDES_PRIMITIVES else len(jtype) == 1 and jtype[0] == "object"
                if field_has_own_properties:
                    reference = None
                    for v in self.keyed_data[key]:
                        if isinstance(v, Node):
                            reference = v
                    final_jtype = reference.schema()

                schema["properties"] = schema["properties"] | {key: final_jtype}
        return schema

    def schema(self) -> dict:
        # TODO: calculate telemetry for each field in our jobject
        #  - sort the properties by key name
        #  - track in COMPARISON to EACH jobject endorsed, how many times that field existed, keep percentage in the end (0.0, 1.0]
        #  - --from-jdump  (load from raw dump of json objects)
        #  - --from-jarray (load from a json array)
        #  - --title (optional argument)
        #  - --description (optional argument)
        #  - --tolerance (optional argument: include fields that were not seen % of the time [0.0, 1.0], by default 0.0; prunes everything that's not everywhere)
        #  - --jobjects-supersede-primitives (optional flag: sets JOBJECT_SUPERSEDES_PRIMITIVES)
        #  - --jarrays-supersede-primitives (optional flag: sets JOBJECT_SUPERSEDES_PRIMITIVES)
        #  - --requireds (optional flag: add a "required" field to each jobject that has required properties)
        #  - --prompt-for-property-description (optional argument: prompt for property description when building final schema)
        #  - add $schema schema keyword draft of JSON Schema standard the schema adheres to
        #  - add $id schema keyword from filename we're loading
        #  - --constraints (optional flag: automatically deduce minimum/maximum constraints for each primitive)
        #      https://json-schema.org/learn/getting-started-step-by-step#define-properties
        #  - --output (optional argument: dump as file w/ filename provided, extension will be .schema.json) }
        #  - --silent (optional argument: do not echo schema to stdout)                                      } mutually inclusive arguments, if --silent is set, --output must be set!

        return self._aggragate_field_types()

    def telemetry(self) -> list[tuple[str, int, int, list[Primitive | Composite]]]:
        """
        To get % of times seen, use round((keyed_endorsements/total_endorsements)*100, 0)
        :return: A list of [path.to.field, keyed_endorsements, total_endorsements, [types]]
        """
        data = []
        for k, v in self.keyed_data.items():
            keyed_endorsement = self.keyed_endorsements[k]

            data.append((
                ".".join([*self.path, k]),
                keyed_endorsement,
                self.total_endorsements,
                v
            ))

            # check for special case: nested endorsements
            nodes_in_values = list(filter(lambda i: isinstance(i, Node), v))
            # data sanity check: this list must never have more than 1 element
            # TODO reenable when this is not failing: assert len(nodes_in_values) <= 1
            if len(nodes_in_values) > 1:
                nested_telemetry = nodes_in_values[0].telemetry()
                for nested_record in nested_telemetry:
                    data.append((
                        nested_record[0],
                        keyed_endorsement + nested_record[1],
                        self.total_endorsements + nested_record[2],
                        nested_record[3]
                    ))
        return data


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


JOBJECT_SUPERSEDES_PRIMITIVES = True


if __name__ == "__main__":
    # TODO add --from-dump <input-file>
    # TODO add --from-array <input-file>
    model: Node = Node()  # root model
    with open(argv[1], "r") as fj_dump:
        while jobj := read_jobj_incrementally(fj_dump):
            model.endorse_jobj(json.loads(jobj))

    print(model)
    print(pprint(model.telemetry()))
    # print(json.dumps(model.schema(), indent=4))
