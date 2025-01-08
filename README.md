# status: stopped/deprecated!
functionality has been covered by the following project https://github.com/alufers/mitmproxy2swagger
~~development has stalled due to lack of time, i WILL return to this when it's even slightly applicable to my day to day (so I can convince myself it's worth it)~~


# jproof
A json-schema generator, based on the strictest interpretation possible from a set of input JSON bodies.

The only requirement for this to work, is that the `$` (root) JSON element *must* be the same across; \ 
otherwise 'likeness' cannot be established between completely foreign objects 
(for example, there is no likeness between a root array and a root object).
