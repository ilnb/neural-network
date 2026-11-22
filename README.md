## README

Neural networks and stuff.

## Build

```sh
cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=Release -G Ninja
cmake --build build
```

## Tests

To run tests, build and run the script:

```sh
./test.sh
```

## Docs

- [Checkpoints](docs/checkpoint.md) — saving and restoring model state
