rm -r -Force buildWindows
cmake -S . -B buildWindows -DCMAKE_BUILD_TYPE=Debug
cmake --build buildWindows --config Debug

rm -r -Force buildWindows
cmake -S . -B buildWindows -DCMAKE_BUILD_TYPE=Release
cmake --build buildWindows --config Release
