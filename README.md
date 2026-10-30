## README

Neural networks and stuff.

## Build

```sh
cmake -S . -B build -G Ninja -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DNN_DTYPE=double -DCMAKE_BUILD_TYPE=Release
cmake --build build
```
