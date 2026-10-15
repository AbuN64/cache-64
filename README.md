# AbuCache64

A Python package that holds the fast paths of an N64 decompilation build. unbake imports it and drives it from its own worker pool.

- `abucache.compile` caches a unit compile by content. The key is the preprocessed bytes, the exact commands and the hash of the tool files. Every ROM version whose input is identical shares one compile.
- `abucache.link` filters the symbol file down to what the object references, builds the linker script and commands, and links with ld and one objcopy per section. Nothing reads the ROM.
- `abucache.store` is the content cache that unbake keeps its step results in.

> This is a personal tool. It is not officially maintained and may change or break without notice.

## Install

```sh
pip install git+https://github.com/AbuN64/cache-64
```

## How the compile cache stays honest

abucache knows no compiler. The caller gives the commands and the tool files. The commands run in a private directory that holds only the input under its own base name. A result that still holds a path of the run is returned but never stored.

## Trace

`abucache.trace()` prints one line per step to stderr. Each line holds the exact argv, and the compile line says whether the cache hit.

## Development

Run `ci/check`.

## License

Released under [GNU GPL version 3 or later](LICENSE).
