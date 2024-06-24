# jproof
A json-schema generator, based on the strictest interpretation possible from a set of input JSON bodies.

The only requirement for this to work, is that the `$` (root) JSON element *must* be the same across; \ 
otherwise 'likeness' cannot be established between completely foreign objects 
(for example, there is no likeness between a root array and a root object).

# json-roulette
A basic json generator.
Usage:
```bash
./json-roulette.py --pretty --size 5 --word-sample-size 15 --composites-size-low 3 --composites-size-high 7
```

### acknowledgements
`words` is a word dump from [this repository](https://github.com/karthikramx/snippable-dictionary)