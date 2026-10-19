# AbuCache64

A Python package that holds the fast paths of an N64 decompilation build. unbake imports it and drives it from its own worker pool.

- `abucache.compile` caches a unit compile by content. The key is the preprocessed bytes, the exact commands and the hash of the tool files. Every ROM version whose input is identical shares one compile.
- `abucache.link` filters the symbol file down to what the object references, builds the linker script and commands, and links with ld and one objcopy per section. Nothing reads the ROM.
- `abucache.store` is the content cache that unbake keeps its step results in. A command only records what it used. The collect of unused entries runs in its own detached process, so no command waits on it.
- `abucache.runtime`, `abucache.keys`, `abucache.files`, `abucache.proc` and `abucache.slice` are the base every tool shares. `runtime` reads the host's one runtime file (cores, workers, memory, named capacities, cache and stage log). `keys` holds the canonical digest, the code identity of a producer and the `Key` of a stored result. `files` writes atomically and locks with a deadline. `proc` runs a child in its own process group with exact accounting. `slice` moves a session command into its systemd slice and leaves services where their unit put them.
- `abucache.stages`, `abucache.memo` and `abucache.events` are the one timing path, the command's memo and the one event writer. `stages` records every command, stage, native child and pool dispatch as a `stage/1` row in the stage log and notes long lock and slot waits in the host's wait log, so a slow or stuck job is seen without per-tool code. `memo` holds a command's pure values and trims them under memory pressure. `events` writes and reads the append-only event feed (schema 2) and ships its JSON schema.
- `abucache.slots` is the host's one set of worker slots: flock files under `$XDG_RUNTIME_DIR/unbake-pool-2` that every command of every tool divides fairly (`cores`, the make `jobserver` and the `run` command body). A holder may name `task_memory`, so its grant never exceeds the tasks the free memory holds, and a named capacity (`[capacities]` in the runtime file) is a second token class for something the host has only a few of. Waits and short shares are stages of the command's tree.
- `abucache.store` also has `Namespace(cache, tool, kind)`, the key space of one tool's results (`"<tool>.<kind>:<key>"`), and `pack`/`unpack`, the stored form of an object. A root with no git history keeps its entries live for the runtime's `keep_seconds`.
- `abucache.require(1)` is the call every tool makes at import: it fails fast when the installed abucache speaks another contract.

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
