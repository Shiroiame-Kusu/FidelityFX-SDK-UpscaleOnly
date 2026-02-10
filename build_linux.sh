rm -rf buildLinux
cmake -S . -B buildLinux -DCMAKE_BUILD_TYPE=Debug
cmake --build buildLinux --config Debug -- -j8

rm -rf buildLinux
cmake -S . -B buildLinux -DCMAKE_BUILD_TYPE=Release
cmake --build buildLinux --config Release -- -j8
