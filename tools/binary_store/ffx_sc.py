#   The MIT License (MIT)
#
#   Copyright (c) 2025 187J3X1-114514
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in
#   all copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#   THE SOFTWARE.
import sys
import os
import threading
from collections import deque
from pathlib import Path
import argparse
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional
import json
import math
import subprocess
import shutil
import hashlib

script_dir = os.path.dirname(os.path.abspath(__file__))


def md5_hash_file(file_path: str) -> str:
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def md5_hash_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def find_include_file(include_file: str, search_paths: List[Path]) -> Path | None:
    local_path = Path(include_file)
    if local_path.exists():
        return local_path.resolve()

    for search_path in search_paths:
        full_path = (search_path / local_path).resolve()
        if full_path.exists():
            return full_path

    return None


def collect_dependencies(
    shader_path: str, include_search_paths: List[Path]
) -> Set[str]:
    dependencies = set()

    if not os.path.exists(shader_path):
        return dependencies

    with open(shader_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:

            stripped = line.lstrip()

            if stripped.startswith("#include"):

                start_quote = -1
                end_quote = -1

                if '"' in stripped:
                    start_quote = stripped.find('"')
                    end_quote = stripped.find('"', start_quote + 1)
                elif "<" in stripped:
                    start_quote = stripped.find("<")
                    end_quote = stripped.find(">", start_quote + 1)

                if start_quote != -1 and end_quote != -1:
                    include_file = stripped[start_quote + 1 : end_quote]
                    include_path = find_include_file(include_file, include_search_paths)

                    if include_path:
                        dependencies.add(str(include_path))

                        sub_deps = collect_dependencies(
                            str(include_path), include_search_paths
                        )
                        dependencies.update(sub_deps)

    return dependencies


def ensure_directory_exists(path: str):

    Path(path).mkdir(parents=True, exist_ok=True)


def normalize_path(path: str) -> str:

    return os.path.normpath(path).replace("\\", "/")


def resolve_executable(
    executable: str, extra_dirs: List[str] | None = None
) -> str | None:

    if executable:
        if os.path.isfile(executable):
            return executable
        if os.name == "nt" and not executable.lower().endswith(".exe"):
            exe_path = executable + ".exe"
            if os.path.isfile(exe_path):
                return exe_path

    search_path = None
    if extra_dirs:
        search_path = os.pathsep.join(extra_dirs)

    if executable:
        resolved = shutil.which(executable, path=search_path)
        if resolved:
            return resolved

    return None


@dataclass
class ShaderResourceInfo:

    name: str
    binding: int
    count: int
    space: int


@dataclass
class ReflectionData:

    constant_buffers: List[ShaderResourceInfo] = field(default_factory=list)
    srv_textures: List[ShaderResourceInfo] = field(default_factory=list)
    uav_textures: List[ShaderResourceInfo] = field(default_factory=list)
    srv_buffers: List[ShaderResourceInfo] = field(default_factory=list)
    uav_buffers: List[ShaderResourceInfo] = field(default_factory=list)
    samplers: List[ShaderResourceInfo] = field(default_factory=list)
    rt_acceleration_structures: List[ShaderResourceInfo] = field(default_factory=list)


@dataclass
class Permutation:

    key: int = 0
    hash_digest: str = ""
    name: str = ""
    header_file_name: str = ""
    defines: List[str] = field(default_factory=list)
    shader_binary: Optional[bytes] = None
    reflection_data: Optional[ReflectionData] = None
    source_path: str = ""
    dependencies: Set[str] = field(default_factory=set)
    identical_to: Optional[int] = None


@dataclass
class PermutationOption:

    definition: str
    values: List[str] = field(default_factory=list)
    num_bits: int = 0
    is_numeric: bool = True
    found_in_shader: bool = False


@dataclass
class LaunchParameters:

    permutation_options: List[PermutationOption] = field(default_factory=list)
    compiler_args: List[str] = field(default_factory=list)
    output_path: str = ""
    input_file: str = ""
    shader_name: str = ""
    compiler: str = ""
    glslang_bin: str = ""
    deps: str = ""
    num_threads: int = 0
    generate_reflection: bool = False
    embed_arguments: bool = False
    print_arguments: bool = False
    disable_logs: bool = False
    debug_compile: bool = False


def parse_permutation_option(definition_str: str) -> PermutationOption:

    if "=" not in definition_str or "{" not in definition_str:
        raise ValueError(f"Invalid permutation option format: {definition_str}")

    equal_pos = definition_str.find("=")
    definition = definition_str.split("=")[0]

    open_brace = definition_str.find("{")
    close_brace = definition_str.find("}")

    if open_brace == -1 or close_brace == -1:
        raise ValueError(f"Invalid permutation option format: {definition_str}")

    values_str = definition_str[open_brace + 1 : close_brace]
    values = [v.strip() for v in values_str.split(",")]

    is_numeric = True
    has_any_numeric = False

    for value in values:
        if value != "-":
            try:
                int(value)
                has_any_numeric = True
            except ValueError:
                is_numeric = False

    if not is_numeric and has_any_numeric:
        raise ValueError(
            f"Permutation option cannot mix numeric and string values: {definition_str}"
        )

    num_bits = int(math.ceil(math.log2(len(values)))) if len(values) > 1 else 1

    return PermutationOption(
        definition=definition, values=values, num_bits=num_bits, is_numeric=is_numeric
    )


def parse_command_line(args: List[str]) -> LaunchParameters:

    parser = argparse.ArgumentParser(
        description="FidelityFX Shader Compiler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("-compiler", choices=["glslang"], help="Compiler to use")
    parser.add_argument("-deps", choices=["gcc", "msvc"], help="Dependency file format")
    parser.add_argument(
        "-D",
        action="append",
        dest="definitions",
        help="Define macro or permutation option",
    )
    parser.add_argument("-e", dest="entry_point", help="Shader entry point")
    parser.add_argument("-name", help="Shader name for output files")
    parser.add_argument("-output", required=True, help="Output directory")
    parser.add_argument(
        "-reflection", action="store_true", help="Generate reflection data"
    )
    parser.add_argument(
        "-embed-arguments",
        action="store_true",
        help="Embed compile arguments in headers",
    )
    parser.add_argument(
        "-print-arguments", action="store_true", help="Print compile arguments"
    )
    parser.add_argument("-disable-logs", action="store_true", help="Disable logging")
    parser.add_argument(
        "-debugcompile", action="store_true", help="Compile with debug info"
    )
    parser.add_argument(
        "-num-threads", type=int, default=0, help="Number of threads to use"
    )
    parser.add_argument("-glslangexe", help="Path to glslangValidator executable")

    parsed, unknown = parser.parse_known_args(args)

    params = LaunchParameters()

    input_file = ""
    for i in range(len(unknown) - 1, -1, -1):
        if unknown[i] and not unknown[i].startswith("-"):
            input_file = unknown.pop(i)
            break

    params.input_file = input_file
    params.compiler = parsed.compiler or ""
    params.deps = parsed.deps or ""
    params.shader_name = parsed.name or ""
    params.output_path = parsed.output
    params.generate_reflection = parsed.reflection
    params.embed_arguments = parsed.embed_arguments
    params.print_arguments = parsed.print_arguments
    params.disable_logs = parsed.disable_logs
    params.debug_compile = parsed.debugcompile
    params.num_threads = parsed.num_threads
    params.glslang_bin = parsed.glslangexe or "glslangValidator"

    if parsed.entry_point:
        params.compiler_args.extend(["-e", parsed.entry_point])

    if parsed.definitions:
        for defn in parsed.definitions:
            if "{" in defn and "}" in defn:

                option = parse_permutation_option(defn)
                params.permutation_options.append(option)
            else:

                if defn.startswith("-D"):
                    params.compiler_args.append(defn)
                else:
                    params.compiler_args.append("-D" + defn)

        # Detect defines duplicated by shell brace expansion (e.g., bash
        # expanding -DNAME={0,1} into -DNAME=0 -DNAME=1) and convert them
        # back into permutation options.
        if not params.permutation_options:
            from collections import OrderedDict

            define_groups = OrderedDict()
            for arg in params.compiler_args:
                if arg.startswith("-D"):
                    name, _, value = arg[2:].partition("=")
                    if name not in define_groups:
                        define_groups[name] = []
                    define_groups[name].append(value)

            duplicated_names = {
                name for name, vals in define_groups.items() if len(vals) > 1
            }

            if duplicated_names:
                new_compiler_args = []
                seen_dup_defines = set()
                for arg in params.compiler_args:
                    if arg.startswith("-D"):
                        name, _, _ = arg[2:].partition("=")
                        if name in duplicated_names:
                            if name not in seen_dup_defines:
                                seen_dup_defines.add(name)
                                values = define_groups[name]
                                is_numeric = all(
                                    v.isdigit() or v == "-" for v in values
                                )
                                num_bits = (
                                    int(math.ceil(math.log2(len(values))))
                                    if len(values) > 1
                                    else 1
                                )
                                params.permutation_options.append(
                                    PermutationOption(
                                        definition=name,
                                        values=values,
                                        num_bits=num_bits,
                                        is_numeric=is_numeric,
                                    )
                                )
                            continue
                    new_compiler_args.append(arg)
                params.compiler_args = new_compiler_args

    i = 0
    while i < len(unknown):
        arg = unknown[i]
        if arg.startswith("-"):
            params.compiler_args.append(arg)
            i += 1

            while i < len(unknown) and not unknown[i].startswith("-"):
                params.compiler_args.append(unknown[i])
                i += 1
        else:

            params.compiler_args.append(arg)
            i += 1

    ensure_output_path(params.output_path)

    return params


def ensure_output_path(output_path: str):

    Path(output_path).mkdir(parents=True, exist_ok=True)


def generate_macro_permutations(
    permutation_options: List[PermutationOption], source_path: str
) -> deque:

    permutations = deque()

    base = Permutation()
    base.source_path = source_path

    sorted_options = sorted(
        permutation_options, key=lambda opt: not opt.found_in_shader
    )

    _generate_recursive(sorted_options, base, permutations, 0, 0)

    return permutations


def _generate_recursive(
    options: List[PermutationOption],
    current: Permutation,
    permutations: deque,
    option_idx: int,
    current_bit: int,
):

    if option_idx >= len(options):
        permutations.append(current)
        return

    option = options[option_idx]

    for value_idx, value in enumerate(option.values):

        new_perm = Permutation()
        new_perm.key = current.key
        new_perm.defines = current.defines.copy()
        new_perm.source_path = current.source_path
        new_perm.identical_to = current.identical_to

        if (
            not option.found_in_shader
            and current.identical_to is None
            and value_idx != 0
        ):
            new_perm.identical_to = new_perm.key

        if value != "-":
            if option.is_numeric:
                new_perm.defines.append(f"-D{option.definition}={value}")
            else:
                new_perm.defines.append(f"-D{value}")

        new_perm.key |= value_idx << current_bit

        _generate_recursive(
            options,
            new_perm,
            permutations,
            option_idx + 1,
            current_bit + option.num_bits,
        )


def find_permutation_options_in_shader(
    shader_path: str,
    include_paths: List[Path],
    permutation_options: List[PermutationOption],
) -> Set[str]:

    search_files = [shader_path]
    searched_files = set()
    num_defs_found = 0

    while search_files and num_defs_found < len(permutation_options):
        current_file = search_files.pop()

        if current_file in searched_files:
            continue

        searched_files.add(current_file)

        try:
            with open(current_file, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    trimmed = line.lstrip()

                    if trimmed.startswith("#include"):

                        start_idx = -1
                        end_idx = -1

                        if '"' in trimmed:
                            start_idx = trimmed.find('"')
                            end_idx = trimmed.find('"', start_idx + 1)
                        elif "<" in trimmed:
                            start_idx = trimmed.find("<")
                            end_idx = trimmed.find(">", start_idx + 1)

                        if start_idx != -1 and end_idx != -1:
                            include_file = trimmed[start_idx + 1 : end_idx]
                            include_path = find_include_file(
                                include_file, include_paths
                            )

                            if include_path:
                                search_files.append(str(include_path))

                    for option in permutation_options:
                        if not option.found_in_shader and option.definition in trimmed:
                            option.found_in_shader = True
                            num_defs_found += 1

        except (OSError, IOError):

            pass

    return searched_files


def write_shader_binary_header(
    permutation: Permutation,
    output_path: str,
    compiler,
    write_mutex,
    embed_arguments: bool = False,
    compiler_args: List[str] = [],
):

    header_path = os.path.join(output_path, f"{permutation.name}.h")
    permutation.header_file_name = f"{permutation.name}.h"

    with open(header_path, "w", encoding="utf-8") as fp:

        fp.write(f"// {permutation.name}.h\n")
        fp.write("// meow\n\n")

        if embed_arguments and compiler_args:
            fp.write("// Compile arguments:\n")
            for arg in compiler_args + permutation.defines:
                fp.write(f"// {arg}\n")
            fp.write("\n")

        if permutation.reflection_data:
            compiler.write_binary_header_reflection_data(fp, permutation, write_mutex)

        binary_size = len(permutation.shader_binary) if permutation.shader_binary else 0
        fp.write(
            f"static const uint32_t g_{permutation.name}_size = {binary_size};\n\n"
        )

        if permutation.shader_binary:
            fp.write(f"static const unsigned char g_{permutation.name}_data[] = {{\n")

            data = permutation.shader_binary
            for i in range(0, len(data), 16):
                chunk = data[i : i + 16]
                hex_values = ", ".join(f"0x{b:02x}" for b in chunk)
                if i + 16 < len(data):
                    hex_values += ","
                fp.write(f"    {hex_values}\n")

            fp.write("};\n")


def write_shader_permutations_header(
    shader_name: str,
    output_path: str,
    unique_permutations: List[Permutation],
    permutation_options: List,
    key_to_index_map: dict,
    compiler,
    generate_reflection: bool,
):

    header_path = os.path.join(output_path, f"{shader_name}_permutations.h")
    for perm in unique_permutations:
        perm.header_file_name = f"{perm.name}.h"
    with open(header_path, "w", encoding="utf-8") as fp:

        for perm in unique_permutations:
            fp.write(f'#include "{perm.header_file_name}"\n')
        fp.write("\n")

        for option in permutation_options:
            if not option.is_numeric:
                enum_name = option.definition
                fp.write(f"typedef enum {enum_name} {{\n")

                for j, value in enumerate(option.values):
                    value_str = value.upper()
                    enum_value = f"OPT_{enum_name.upper()}_{value_str} = {j}"
                    if j == len(option.values) - 1:
                        fp.write(f"    {enum_value}\n")
                    else:
                        fp.write(f"    {enum_value},\n")

                fp.write(f"}} {enum_name};\n\n")

        union_name = f"{shader_name}_PermutationKey"
        fp.write(f"typedef union {union_name} {{\n")
        fp.write("    struct {\n")

        for option in permutation_options:
            fp.write(f"        uint32_t {option.definition} : {option.num_bits};\n")

        fp.write("    };\n")
        fp.write("    uint32_t index;\n")
        fp.write(f"}} {union_name};\n\n")

        fp.write(f"typedef struct {shader_name}_PermutationInfo {{\n")
        fp.write("    const uint32_t       blobSize;\n")
        fp.write("    const unsigned char* blobData;\n\n")

        if generate_reflection:
            compiler.write_permutation_header_reflection_struct_members(fp)

        fp.write(f"}} {shader_name}_PermutationInfo;\n\n")

        used_bits = sum(opt.num_bits for opt in permutation_options)
        total_possible = 2**used_bits

        fp.write(f"static const uint32_t g_{shader_name}_IndirectionTable[] = {{\n")
        for i in range(total_possible):
            index = key_to_index_map.get(i, 0)
            fp.write(f"    {index},\n")
        fp.write("};\n\n")

        if unique_permutations:
            fp.write(
                f"static const {shader_name}_PermutationInfo "
                f"g_{shader_name}_PermutationInfo[] = {{\n"
            )

            for perm in unique_permutations:
                fp.write("    {\n")
                fp.write(f"        g_{perm.name}_size,\n")
                fp.write(f"        g_{perm.name}_data,\n")

                if generate_reflection:
                    fp.write("        ")
                    compiler.write_permutation_header_reflection_data(fp, perm)
                    fp.write("\n")

                fp.write("    },\n")

            fp.write("};\n")


def write_depfile_gcc(shader_name: str, output_path: str, dependencies: set):

    header_file = os.path.join(output_path, f"{shader_name}_permutations.h")
    depfile_path = header_file + ".d"

    with open(depfile_path, "w", encoding="utf-8") as fp:
        fp.write(f"{header_file}:")
        for dep in sorted(dependencies):
            fp.write(f" {dep}")
        fp.write("\n")


def print_help():
    print("Command line syntax:")
    print(f"  python ffx_sc.py [Options] <InputFile>")
    print()
    print("Options:")
    print("  -D<Name>")
    print("    Define a macro that is defined in all shader permutations.")
    print()
    print("  -D<Name>={<Value1>,<Value2>,...}")
    print("    Declare a shader option that will generate permutations.")
    print("    Use a '-' to define a permutation where no macro is defined.")
    print()
    print("  -output=<Path>")
    print("    Path to where the shader permutations should be output to.")
    print()
    print("  -name=<Name>")
    print("    The name used for prefixing variables in the generated headers.")
    print("    Uses the file name by default.")
    print()
    print("  -compiler=<Compiler>")
    print("    Select the compiler to generate permutations (glslang).")
    print()
    print("  -glslangexe=<Path>")
    print("    Path to the glslangValidator executable to use.")
    print()
    print("  -num-threads=<Num>")
    print("    Number of threads to use for generating shaders.")
    print("    Uses max available CPU threads by default.")
    print()
    print("  -reflection")
    print("    Generate header containing reflection data.")
    print()
    print("  -embed-arguments")
    print("    Write the compile arguments into the headers.")
    print()
    print("  -print-arguments")
    print("    Print the compile arguments for each permutation.")
    print()
    print("  -disable-logs")
    print("    Prevent logging of compile warnings and errors.")
    print()
    print("  -deps=<Format>")
    print("    Dump depfile which recorded include dependencies (gcc or msvc).")
    print()
    print("  -debugcompile")
    print("    Compile shader with debug information.")
    print()
    print("Additional compiler-specific arguments:")
    print("  -e <entry>        Entry point name")
    print("  -S <stage>        Shader stage")
    print("  -I <path>         Include path")
    print("  --target-env <env> Target environment (e.g. vulkan1.2)")
    print("  -Os               Optimize for size")

class GLSLCompiler():

    @staticmethod
    def _find_tool(name: str, test_arg: str = "--version") -> str:
        """Find a tool executable, preferring bundled > VULKAN_SDK > system PATH."""
        VULKAN_SDK_BUILD_DIR = os.environ.get("VULKAN_SDK_BUILD_DIR")
        VULKAN_SDK_DIR = os.environ.get("VULKAN_SDK")
        extra_str = ".exe" if os.name == "nt" else ""

        candidates = []

        # 1. Try local bundled executable first (version-matched to this SDK)
        if os.name == "nt":
            candidates.append(os.path.join(script_dir, "windows", name + ".exe"))
        else:
            candidates.append(os.path.join(script_dir, "linux", name))

        # 2. Try VULKAN_SDK_BUILD_DIR
        if VULKAN_SDK_BUILD_DIR:
            candidates.append(
                os.path.join(VULKAN_SDK_BUILD_DIR, "tools", "bin", name + extra_str)
            )

        # 3. Try VULKAN_SDK
        if VULKAN_SDK_DIR:
            candidates.append(
                os.path.join(VULKAN_SDK_DIR, "Bin", name + extra_str)
            )

        # 4. Try system PATH
        candidates.append(name)

        for candidate in candidates:
            try:
                # Ensure execute permission for bundled Linux binaries
                if os.path.isfile(candidate) and os.name != "nt":
                    if not os.access(candidate, os.X_OK):
                        os.chmod(candidate, os.stat(candidate).st_mode | 0o111)

                result = subprocess.run(
                    [candidate, test_arg], capture_output=True
                )
                if result.returncode == 0:
                    return candidate
            except Exception:
                pass

        raise FileNotFoundError(
            f"{name} executable not found. Please install Vulkan SDK or "
            f"ensure {name} is in PATH."
        )

    def __init__(
        self,
        glslang_bin: str,
        shader_path: str,
        shader_name: str,
        shader_file_name: str,
        output_path: str,
        disable_logs: bool,
        debug_compile: bool,
    ):

        self.shader_path = shader_path
        self.shader_name = shader_name
        self.shader_file_name = shader_file_name
        self.output_path = output_path
        self.disable_logs = disable_logs
        self.debug_compile = debug_compile

        self.glslang_bin = self._find_tool("glslangValidator", "--version")
        self.spirv_cross_bin = self._find_tool("spirv-cross", "--help")
        self.spirv_reflect_bin = self._find_tool("spirv-reflect", "--help")

        self.shader_dependencies = set()
        self.shader_dependencies_collected = False

        temp_dir = os.path.join(output_path, shader_name + "_temp")
        ensure_directory_exists(temp_dir)
        self.temp_dir = temp_dir

    def __del__(self):

        import shutil

        if hasattr(self, "temp_dir") and os.path.exists(self.temp_dir):
            pass  # shutil.rmtree(self.temp_dir, ignore_errors=True)

    def compile(
        self,
        permutation: Permutation,
        arguments: List[str],
        write_mutex: threading.Lock,
    ) -> bool:

        errors = []

        cmd = [self.glslang_bin]

        if self.debug_compile:
            cmd.extend(["-g", "-gVS", "-Od"])

        cmd.append("-V")

        include_search_paths = []
        i = 0
        while i < len(arguments):
            arg = arguments[i]

            if not arg or not arg.strip():
                i += 1
                continue

            if arg.startswith("-I"):
                include_search_paths.append(Path(arg[2:]))
                cmd.append(arg)
                i += 1
            elif arg == "-e":

                cmd.append(arg)
                if i + 1 < len(arguments):
                    i += 1
                    cmd.append(arguments[i])
                i += 1
            else:
                cmd.append(arg)
                i += 1

        with write_mutex:
            if not self.shader_dependencies_collected:
                self.shader_dependencies_collected = True
                self.shader_dependencies = collect_dependencies(
                    self.shader_path, include_search_paths
                )

        temp_file = os.path.join(self.temp_dir, f"{permutation.key}.spv")

        cmd.extend([self.shader_path, "-o", temp_file])

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )

            output = result.stderr + result.stdout
            for line in output.split("\n"):
                if line.strip() and (
                    "error" in line.lower() or "warning" in line.lower()
                ):

                    line_num = -1
                    if ":" in line:
                        parts = line.split(":")
                        for part in parts:
                            if part.strip().isdigit():
                                line_num = int(part.strip())
                                break
                    errors.append({"error": line, "lineNumber": line_num})

            succeeded = result.returncode == 0

        except Exception as e:
            errors.append({"error": str(e), "lineNumber": -1})
            print(f"Exception during compilation: {e}")
            succeeded = False

        if len(errors) > 0:
            with write_mutex:
                print(
                    f"{self.shader_file_name}[{permutation.key}]",
                    file=__import__("sys").stderr,
                )
                for err in errors:
                    if err["lineNumber"] != -1:
                        print(
                            f"  Line {err['lineNumber']}: {err['error']}",
                            file=__import__("sys").stderr,
                        )
                    else:
                        print(f"  {err['error']}", file=__import__("sys").stderr)

        if succeeded and os.path.exists(temp_file):

            with open(temp_file, "rb") as f:
                spirv_data = f.read()

            spirv_hash = md5_hash_bytes(spirv_data)
            permutation.hash_digest = spirv_hash
            permutation.name = f"{self.shader_name}_{spirv_hash}"

            hash_file = os.path.join(self.output_path, f"{permutation.name}.spv")

            try:
                if os.path.exists(hash_file):
                    pass  # os.remove(hash_file)
                os.rename(temp_file, hash_file)
            except Exception as e:
                pass

            permutation.shader_binary = spirv_data

        permutation.dependencies = self.shader_dependencies

        return succeeded
    
    def extract_reflection_data_from_spirv_reflect(
        self, spirv_file: str, spirv_reflect_bin: str
    ) -> Dict:
        reflection_data = {}
        import yaml
        try:
            try:
                result = subprocess.run(
                    [
                        spirv_reflect_bin,
                        spirv_file,
                        "-y"
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                
            except Exception as e:
                if not self.disable_logs:
                    print(
                        f"Failed to run spirv-reflect: {e}", file=__import__("sys").stderr
                    )
                return {}
            if result.returncode != 0:
                return {}
            yaml_output = result.stdout
            count = 0
            while True:
                try:
                    if os.name != "nt":
                        import time
                        time.sleep(0.1)
                    data = yaml.safe_load(yaml_output)
                    return data["all_descriptor_bindings"]
                    break
                except Exception as e:
                    if count >= 5:
                        raise e
                    count += 1
        except Exception as e:
            if not self.disable_logs:
                print(
                    f"Failed to extract reflection: {e} {spirv_file}",
                    file=__import__("sys").stderr,
                )
        pass

    def extract_reflection_data(
        self, permutation: Permutation, spirv_cross_bin: str,spirv_reflect_bin :str
    ) -> bool:

        reflection = ReflectionData()

        spirv_file = os.path.join(self.output_path, f"{permutation.name}.spv")

        if not os.path.exists(spirv_file):
            return False

        json_file = os.path.join(self.temp_dir, f"{permutation.name}.json")

        try:

            try:
                result = subprocess.run(
                    [
                        spirv_cross_bin,
                        "-V",
                        spirv_file,
                        "--reflect",
                        "--output",
                        json_file,
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            except Exception as e:
                if not self.disable_logs:
                    print(
                        f"Failed to run spirv-cross: {e}", file=__import__("sys").stderr
                    )
                return False

            if result.returncode != 0 or not os.path.exists(json_file):
                return False

            with open(json_file, "r", encoding="utf-8") as f:
                count = 0
                while True:
                    try:
                        if os.name != "nt":
                            import time
                            time.sleep(0.1)
                        data = json.load(f)
                        
                        break
                    except Exception as e:
                        if count >= 5:
                            raise e
                        count += 1

            
            #ubo数据有问题，得用spirv-reflect重新提取
            patch = self.extract_reflection_data_from_spirv_reflect(spirv_file, spirv_reflect_bin)
            """
            {
                "type" : "_398",
                "name" : "cbFSR3UPSCALER_t", #!
                "block_size" : 148,
                "set" : 0,     #!
                "binding" : 12 #!
            }
            """
            data["ubos"] = []
            for patch_resource in patch:
                if int(patch_resource["resource_type"]) == 2:
                    data["ubos"].append({
                        "name": patch_resource["name"],
                        "binding": patch_resource["binding"],
                        "set": patch_resource["set"],
                    })
            self._extract_resources(data, "ubos", reflection.constant_buffers)
            self._extract_resources(data, "separate_images", reflection.srv_textures)
            self._extract_resources(data, "images", reflection.uav_textures)
            self._extract_resources(data, "ssbos", reflection.srv_buffers)
            self._extract_resources(data, "storage_buffers", reflection.uav_buffers)
            self._extract_resources(data, "separate_samplers", reflection.samplers)
            #用不到
            #self._extract_resources(data, "acceleration_structures", reflection.rt_acceleration_structures)

        except Exception as e:
            if not self.disable_logs:
                print(
                    f"Failed to extract reflection: {e} {json_file}",
                    file=__import__("sys").stderr,
                )
            return False

        permutation.reflection_data = reflection
        return True

    def _extract_resources(
        self, data: dict, key: str, resource_list: List[ShaderResourceInfo]
    ):

        if key not in data:
            return

        for res in data[key]:
            name = res["name"]
            resource_list.append(
                ShaderResourceInfo(
                    name=name,
                    binding=res.get("binding", 0),
                    count=1,
                    space=res.get("set", 0),
                )
            )

    def write_binary_header_reflection_data(
        self, fp, permutation: Permutation, write_mutex: threading.Lock
    ):

        if not permutation.reflection_data:
            return

        reflection = permutation.reflection_data

        self._write_resource_info(
            fp, permutation.name, reflection.constant_buffers, "CBV"
        )
        self._write_resource_info(
            fp, permutation.name, reflection.srv_textures, "TextureSRV"
        )
        self._write_resource_info(
            fp, permutation.name, reflection.uav_textures, "TextureUAV"
        )
        self._write_resource_info(
            fp, permutation.name, reflection.srv_buffers, "BufferSRV"
        )
        self._write_resource_info(
            fp, permutation.name, reflection.uav_buffers, "BufferUAV"
        )
        self._write_resource_info(fp, permutation.name, reflection.samplers, "Sampler")
        self._write_resource_info(
            fp,
            permutation.name,
            reflection.rt_acceleration_structures,
            "RTAccelerationStructure",
        )

    def _write_resource_info(
        self,
        fp,
        permutation_name: str,
        resources: List[ShaderResourceInfo],
        resource_type: str,
    ):

        if not resources:
            return

        resources = sorted(resources, key=lambda r: r.binding)

        names = ", ".join(f'"{r.name}"' for r in resources)
        fp.write(
            f"static const char* g_{permutation_name}_{resource_type}ResourceNames[] = "
            f"{{ {names} }};\n"
        )

        bindings = ", ".join(str(r.binding) for r in resources)
        fp.write(
            f"static const uint32_t g_{permutation_name}_{resource_type}ResourceBindings[] = "
            f"{{ {bindings} }};\n"
        )

        counts = ", ".join(str(r.count) for r in resources)
        fp.write(
            f"static const uint32_t g_{permutation_name}_{resource_type}ResourceCounts[] = "
            f"{{ {counts} }};\n"
        )

        spaces = ", ".join(str(r.space) for r in resources)
        fp.write(
            f"static const uint32_t g_{permutation_name}_{resource_type}ResourceSets[] = "
            f"{{ {spaces} }};\n"
        )

        fp.write("\n")

    def write_permutation_header_reflection_struct_members(self, fp):

        resource_definitions = [
            ("SRVTextures", "srvTexture"),
            ("UAVTextures", "uavTexture"),
            ("SRVBuffers", "srvBuffer"),
            ("UAVBuffers", "uavBuffer"),
            ("Samplers", "sampler"),
            ("RTAccelerationStructures", "rtAccelerationStructure"),
        ]

        fp.write(f"    const uint32_t  numConstantBuffers;\n")
        fp.write(f"    const char**    constantBufferNames;\n")
        fp.write(f"    const uint32_t* constantBufferBindings;\n")
        fp.write(f"    const uint32_t* constantBufferCounts;\n")
        fp.write(f"    const uint32_t* constantBufferSpaces;\n")
        fp.write("\n")

        for count_name, member_prefix in resource_definitions:
            fp.write(f"    const uint32_t  num{count_name};\n")
            fp.write(f"    const char**    {member_prefix}Names;\n")
            fp.write(f"    const uint32_t* {member_prefix}Bindings;\n")
            fp.write(f"    const uint32_t* {member_prefix}Counts;\n")
            fp.write(f"    const uint32_t* {member_prefix}Spaces;\n")
            fp.write("\n")

    def write_permutation_header_reflection_data(self, fp, permutation: Permutation):

        if not permutation.reflection_data:

            fp.write("0, 0, 0, 0, 0, " * 7)
            return

        reflection = permutation.reflection_data

        self._write_resource_initializer(
            fp, permutation.name, reflection.constant_buffers, "CBV"
        )
        self._write_resource_initializer(
            fp, permutation.name, reflection.srv_textures, "TextureSRV"
        )
        self._write_resource_initializer(
            fp, permutation.name, reflection.uav_textures, "TextureUAV"
        )
        self._write_resource_initializer(
            fp, permutation.name, reflection.srv_buffers, "BufferSRV"
        )
        self._write_resource_initializer(
            fp, permutation.name, reflection.uav_buffers, "BufferUAV"
        )
        self._write_resource_initializer(
            fp, permutation.name, reflection.samplers, "Sampler"
        )
        self._write_resource_initializer(
            fp,
            permutation.name,
            reflection.rt_acceleration_structures,
            "RTAccelerationStructure",
        )

    def _write_resource_initializer(
        self,
        fp,
        permutation_name: str,
        resources: List[ShaderResourceInfo],
        resource_type: str,
    ):

        if resources:
            fp.write(f"{len(resources)}, ")
            fp.write(f"g_{permutation_name}_{resource_type}ResourceNames, ")
            fp.write(f"g_{permutation_name}_{resource_type}ResourceBindings, ")
            fp.write(f"g_{permutation_name}_{resource_type}ResourceCounts, ")
            fp.write(f"g_{permutation_name}_{resource_type}ResourceSets, ")
        else:
            fp.write("0, 0, 0, 0, 0, ")


class Application:

    def __init__(self, params: LaunchParameters):

        self.params = params
        self.compiler = None
        self.macro_permutations = deque()
        self.unique_permutations: List[Permutation] = []
        self.read_mutex = threading.Lock()
        self.write_mutex = threading.Lock()
        self.last_permutation_index = 0
        self.key_to_index_map: Dict[int, int] = {}
        self.hash_to_index_map: Dict[str, int] = {}
        self.shader_file_name = ""
        self.shader_name = ""

    def process(self):

        self._open_source_file()

        self.macro_permutations = generate_macro_permutations(
            self.params.permutation_options, self.params.input_file
        )

        predicted_duplicates = sum(
            1 for p in self.macro_permutations if p.identical_to is not None
        )

        total_permutations = len(self.macro_permutations)

        if self.params.num_threads == 0:
            import multiprocessing

            self.params.num_threads = multiprocessing.cpu_count()

        self.params.num_threads = min(
            self.params.num_threads, total_permutations - predicted_duplicates
        )

        print(f"{self.shader_file_name}")

        threads = []
        for i in range(1):
            thread = threading.Thread(target=self.process_permutations_method)
            thread.start()
            threads.append(thread)

        self.process_permutations_method()

        for thread in threads:
            thread.join()

        self.write_shader_permutations_header_method()

        if self.params.deps == "gcc":
            self.dump_depfile_gcc_method()
        elif self.params.deps == "msvc":
            print("MSVC depfile not implemented yet.", file=sys.stderr)

    def _open_source_file(self):

        input_path = Path(self.params.input_file)
        self.shader_file_name = input_path.name

        if self.params.shader_name:
            self.shader_name = self.params.shader_name
        else:
            self.shader_name = input_path.stem

        if not self.params.compiler:

            if input_path.suffix.lower() == ".glsl":
                self.params.compiler = "glslang"
            else:
                raise RuntimeError(
                    "Unknown shader source file extension. "
                    "Please use the -compiler option."
                )

        if self.params.compiler == "glslang":
            self.compiler = GLSLCompiler(
                self.params.glslang_bin,
                self.params.input_file,
                self.shader_name,
                self.shader_file_name,
                self.params.output_path,
                self.params.disable_logs,
                self.params.debug_compile,
            )
        else:
            raise RuntimeError(f"Unknown compiler: {self.params.compiler}")

        include_paths = []
        i = 0
        while i < len(self.params.compiler_args):
            arg = self.params.compiler_args[i]
            if arg == "-I" and i + 1 < len(self.params.compiler_args):
                include_paths.append(Path(self.params.compiler_args[i + 1]))
                i += 1
            elif arg.startswith("-I"):
                include_paths.append(Path(arg[2:]))
            i += 1

        if self.params.permutation_options:
            find_permutation_options_in_shader(
                self.params.input_file, include_paths, self.params.permutation_options
            )

    def process_permutations_method(self):

        while True:

            with self.read_mutex:
                if not self.macro_permutations:
                    break
                permutation = self.macro_permutations.pop()

            self.compile_permutation_method(permutation)

    def compile_permutation_method(self, permutation):

        if permutation.identical_to is not None:
            with self.write_mutex:

                if permutation.identical_to in self.key_to_index_map:
                    index = self.key_to_index_map[permutation.identical_to]
                    self.key_to_index_map[permutation.key] = index
                    return
                else:

                    pass

            with self.read_mutex:
                self.macro_permutations.appendleft(permutation)
            return

        args = []
        args.extend(permutation.defines)
        args.extend(self.params.compiler_args)

        if self.params.print_arguments:
            self.print_permutation_arguments_method(permutation)

        if not self.compiler.compile(permutation, args, self.write_mutex):
            raise RuntimeError(f"Failed to compile shader: {permutation.source_path}")

        if self.params.generate_reflection:
            self.compiler.extract_reflection_data(
                permutation, self.compiler.spirv_cross_bin, self.compiler.spirv_reflect_bin
            )

        should_write = False

        with self.write_mutex:
            if permutation.hash_digest not in self.hash_to_index_map:
                should_write = True

                self.hash_to_index_map[permutation.hash_digest] = (
                    self.last_permutation_index
                )
                self.last_permutation_index += 1

                unique_perm = Permutation()
                unique_perm.key = permutation.key
                unique_perm.hash_digest = permutation.hash_digest
                unique_perm.name = permutation.name
                unique_perm.defines = permutation.defines
                unique_perm.reflection_data = permutation.reflection_data
                unique_perm.source_path = permutation.source_path
                unique_perm.dependencies = permutation.dependencies
                unique_perm.identical_to = permutation.identical_to

                self.unique_permutations.append(unique_perm)

            self.key_to_index_map[permutation.key] = self.hash_to_index_map[
                permutation.hash_digest
            ]

        if should_write:
            write_shader_binary_header(
                permutation,
                self.params.output_path,
                self.compiler,
                self.write_mutex,
                self.params.embed_arguments,
                self.params.compiler_args + permutation.defines,
            )

    def print_permutation_arguments_method(self, permutation):

        with self.write_mutex:
            print(f"\nPermutation {permutation.key}:")

            if self.params.generate_reflection:
                print("-reflection ", end="")

            for arg in self.params.compiler_args:
                print(f"{arg} ", end="")

            for define in permutation.defines:
                print(f"{define} ", end="")

            print(f"-output={self.params.output_path}")
            print()

    def write_shader_permutations_header_method(self):

        if not self.unique_permutations:
            raise RuntimeError("No shader permutations generated due to errors!")

        write_shader_permutations_header(
            self.shader_name,
            self.params.output_path,
            self.unique_permutations,
            self.params.permutation_options,
            self.key_to_index_map,
            self.compiler,
            self.params.generate_reflection,
        )

    def dump_depfile_gcc_method(self):

        if not self.unique_permutations:
            raise RuntimeError("No shader permutations generated due to errors!")

        all_dependencies = set()
        for perm in self.unique_permutations:
            all_dependencies.update(perm.dependencies)

        write_depfile_gcc(self.shader_name, self.params.output_path, all_dependencies)


def main():

    try:

        if len(sys.argv) <= 1 or "--help" in sys.argv or "-h" in sys.argv:
            print_help()
            return 0

        params = parse_command_line(sys.argv[1:])

        app = Application(params)
        app.process()

        return 0

    except Exception as e:
        print(f"ffx_sc failed: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
