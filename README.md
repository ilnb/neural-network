## README

Neural networks and stuff.

## Build

```sh
cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=Release -DNN_DTYPE=double -G Ninja
cmake --build build
```
