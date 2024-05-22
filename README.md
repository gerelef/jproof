# pyjama-proof
A json-schema generator, based on the strictest interpretation possible from a set of input JSON bodies.

The only requirement for this to work, is that the `$` (root) JSON element *must* be the same across; otherwise 'likeness' cannot be established between completely foreign objects (for example, there is no likeness between an array and an object).

## flags
```
--generalize                        -> will try to generate $refs for seemingly common values
--no-strict-numbers                 -> skip trying to predict :integer types instead of :number
--number-skip=minimum:maximum:      -> skip setting all number minimum-maximums
--string-skip=minlength:maxlength   -> skip setting all string minimum-maximums
--array-skip=minimum:maximum:unique -> skip setting all array minimum/maximums-uniques
--object-skip=minimum:maximum       -> skip setting minimum-maximum property count
```
