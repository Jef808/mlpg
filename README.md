To configure and build:
```sh
conan install --install-folder=build --build=missing .
cmake -G Ninja -S . -B build
cmake --build build
```
