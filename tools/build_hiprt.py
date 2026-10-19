#!/usr/bin/env python3
"""Build the HIPRT SDK payload from public source, without release binaries."""

import argparse
import datetime
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

SOURCE_URL = "https://github.com/GPUOpen-LibrariesAndSDKs/HIPRT.git"
WINDOWS = sys.platform == "win32"


def run(args, cwd, env, capture=False):
    args = [str(arg) for arg in args]
    print("+ " + subprocess.list2cmdline(args), flush=True)
    result = subprocess.run(args, cwd=str(cwd), env=env, check=True,
                            stdout=subprocess.PIPE if capture else None,
                            text=True)
    return result.stdout.strip() if capture else ""


def git(args, cwd, env, capture=False):
    # The library build does not use LFS test assets or bundled third-party DLLs.
    return run(["git", "-c", "filter.lfs.required=false", "-c", "filter.lfs.process=",
                "-c", "filter.lfs.smudge="] + args, cwd, env, capture)


def require_file(path):
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError("Required file is missing or empty: " + str(path))
    return path


def sdk_path(value, variable, fallback):
    result = value or os.environ.get(variable) or fallback
    return Path(result).expanduser().resolve() if result else None


def vs2022(env, root):
    locator = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Microsoft Visual Studio/Installer/vswhere.exe"
    require_file(locator)
    result = run([locator, "-latest", "-products", "*", "-version", "[17.0,18.0)",
                  "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                  "-property", "installationPath"], root, env, True)
    if not result:
        raise RuntimeError("Install Visual Studio 2022 / Build Tools with Desktop development with C++.")
    return Path(result)


def resolve_revision(ref, root, env):
    if re.fullmatch(r"[0-9a-fA-F]{40}", ref):
        return ref.lower()
    patterns = [ref] if ref.startswith("refs/") or ref == "HEAD" else ["refs/heads/" + ref, "refs/tags/" + ref]
    patterns += [name + "^{}" for name in patterns if name.startswith("refs/tags/")]
    output = git(["ls-remote", SOURCE_URL] + patterns, root, env, True)
    matches = {}
    for line in output.splitlines():
        sha, name = line.split()
        matches[name] = sha
    revisions = {matches.get(name + "^{}", sha) for name, sha in matches.items()}
    if len(revisions) != 1:
        raise RuntimeError("HIPRT --ref must identify one branch, tag or full commit SHA: " + ref)
    return revisions.pop()


def checkout(source, revision, env):
    if not source.exists():
        source.mkdir(parents=True)
        git(["init"], source, env)
        git(["remote", "add", "origin", SOURCE_URL], source, env)
    if not (source / ".git").is_dir():
        raise RuntimeError("The managed source directory is not a Git checkout: " + str(source))
    if git(["remote", "get-url", "origin"], source, env, True) != SOURCE_URL:
        raise RuntimeError("Unexpected origin in managed HIPRT checkout; use another --build-dir.")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(source), env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    if head.returncode or head.stdout.strip() != revision:
        if head.returncode == 0:
            raise RuntimeError("Unexpected revision in managed checkout; use another --build-dir.")
        git(["fetch", "--depth", "1", "origin", revision], source, env)
        git(["checkout", "--detach", revision], source, env)
    git(["submodule", "update", "--init", "--recursive"], source, env)
    changed = git(["diff", "--name-only", "HEAD"], source, env, True).splitlines()
    if changed:
        raise RuntimeError("Modified sources in managed checkout; preserve your changes and use another --build-dir.")


def install(root, build, headers, artifacts, hip, use_amd, metadata):
    destination = root / "hiprt"
    platform = "win" if WINDOWS else "linux64"
    stage = Path(tempfile.mkdtemp(prefix="install-", dir=str(build)))
    if destination.exists():
        # Preserve README, the other platform and user-added files. Replace this
        # platform completely so stale release binaries cannot hide a failure.
        shutil.copytree(destination, stage, dirs_exist_ok=True,
                        ignore=lambda folder, names: [platform] if Path(folder) == destination else [])
    for header in headers.glob("*.h"):
        shutil.copy2(require_file(header), stage / header.name)
    binaries = stage / platform
    binaries.mkdir()
    for artifact in artifacts:
        shutil.copy2(require_file(artifact), binaries / artifact.name)
    if WINDOWS and use_amd:
        dependencies = list((hip / "bin").glob("hiprtc*.dll")) + list((hip / "bin").glob("amd_comgr*.dll"))
        for dependency in dependencies:
            shutil.copy2(require_file(dependency), binaries / dependency.name)
    (binaries / "source-build.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    backup = build / ("hiprt-backup-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    if destination.exists():
        destination.rename(backup)
        print("Previous hiprt directory saved in " + str(backup), flush=True)
    try:
        stage.rename(destination)
    except OSError:
        if backup.exists():
            backup.rename(destination)
        raise
    print("Installed source-built HIP RT in " + str(destination), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", default="main", help="HIPRT branch, tag or full commit SHA (default: main)")
    parser.add_argument("--hip-path", help="HIP SDK / ROCm root (default: HIP_PATH, then /opt/rocm on Linux)")
    parser.add_argument("--backend", choices=["amd", "cuda", "both"], default="amd", help="GPU backend (default: amd)")
    parser.add_argument("--cuda-path", help="CUDA toolkit root for cuda/both (default: CUDA_PATH, then /usr/local/cuda on Linux)")
    parser.add_argument("--jobs", type=int, default=min(os.cpu_count() or 1, 16), help="Parallel CMake build jobs; GPU compilation uses HIPRT defaults")
    parser.add_argument("--build-dir", type=Path, help="Managed build directory (default: build/hiprt-source/BACKEND/REVISION)")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if sys.platform not in ("win32", "linux"):
        parser.error("Only Windows and Linux are supported")
    root = Path(__file__).resolve().parent.parent
    env = os.environ.copy()
    env["GIT_LFS_SKIP_SMUDGE"] = "1"
    env["PYTHON_BIN"] = '"' + sys.executable + '"' if WINDOWS else sys.executable
    use_amd, use_cuda = args.backend != "cuda", args.backend != "amd"
    hip = sdk_path(args.hip_path, "HIP_PATH", None if WINDOWS else "/opt/rocm")
    cuda = sdk_path(args.cuda_path, "CUDA_PATH", "/usr/local/cuda" if not WINDOWS else None)
    for tool in ("git", "cmake"):
        if not shutil.which(tool):
            raise RuntimeError("Install " + tool + " and add it to PATH.")
    if use_amd:
        if hip is None:
            raise RuntimeError("Set HIP_PATH or pass --hip-path to your HIP SDK installation.")
        require_file(hip / "bin" / ("hipcc.exe" if WINDOWS else "hipcc"))
        env["HIP_PATH"] = str(hip)
        env["PATH"] = str(hip / "bin") + os.pathsep + env.get("PATH", "")
    if use_cuda:
        if cuda is None:
            raise RuntimeError("Set CUDA_PATH or pass --cuda-path to your CUDA toolkit.")
        require_file(cuda / "bin" / ("nvcc.exe" if WINDOWS else "nvcc"))
        env["CUDA_PATH"] = str(cuda)
        env["PATH"] = str(cuda / "bin") + os.pathsep + env.get("PATH", "")
    visual_studio = vs2022(env, root) if WINDOWS else None
    if WINDOWS:
        # Both hipcc and nvcc need the VS 2022 host environment; otherwise Clang
        # can silently pick an incompatible newer Visual Studio installation.
        command = 'call "' + str(visual_studio / "Common7/Tools/VsDevCmd.bat") + '" -arch=x64 >nul && set'
        print("Configuring the Visual Studio 2022 x64 environment", flush=True)
        output = subprocess.run(command, shell=True, cwd=str(root), env=env,
                                stdout=subprocess.PIPE, text=True, check=True).stdout
        env.update(line.split("=", 1) for line in output.splitlines() if "=" in line and not line.startswith("="))
        env["HIPRT_VS_VERSION"] = "2022"
    revision = resolve_revision(args.ref, root, env)
    print("Building HIPRT " + args.ref + " at " + revision, flush=True)
    build = (args.build_dir or root / "build/hiprt-source" / args.backend / revision[:12]).resolve()
    if build == root or root / "hiprt" in (build, *build.parents):
        raise RuntimeError("--build-dir must be outside the hiprt installation directory and cannot be the repository root.")
    # Install staging must share the destination's filesystem for the final rename.
    if build.drive != root.drive:
        raise RuntimeError("--build-dir must be on the same filesystem as HIPRTSDK.")
    source = build / "src"
    build.mkdir(parents=True, exist_ok=True)
    if build.stat().st_dev != root.stat().st_dev:
        raise RuntimeError("--build-dir must be on the same filesystem as HIPRTSDK.")
    checkout(source, revision, env)
    cmake_args = ["cmake", "-S", source, "-B", build / "cmake", "-DCMAKE_BUILD_TYPE=Release",
                  "-DNO_UNITTEST=ON", "-DBITCODE=ON", "-DBAKE_KERNEL=OFF",
                  "-DPRECOMPILE=" + ("ON" if use_amd else "OFF"),
                  "-DBAKE_COMPILED_KERNEL=OFF", "-DGENERATE_BAKE_KERNEL=ON",
                  "-DFORCE_DISABLE_CUDA=" + ("OFF" if use_cuda else "ON")]
    if use_amd:
        cmake_args += ["-DPython3_EXECUTABLE=" + sys.executable]
    if hip:
        cmake_args += ["-DHIP_PATH=" + str(hip)]
    if WINDOWS:
        cmake_args += ["-G", "Visual Studio 17 2022", "-A", "x64"]
    run(cmake_args, root, env)
    # Upstream's bake scripts don't propagate Python errors. Verify their outputs.
    for generated in (source / "hiprt/cache/Kernels.h", source / "hiprt/cache/KernelArgs.h",
                      source / "contrib/Orochi/ParallelPrimitives/cache/Kernels.h"):
        if require_file(generated).stat().st_size < 100:
            raise RuntimeError("Kernel source generation failed: " + str(generated))
    run(["cmake", "--build", build / "cmake", "--config", "Release", "--parallel", args.jobs], root, env)
    if not use_amd:
        # CMake's PRECOMPILE target always includes AMD. The official script
        # also exposes a CUDA-only mode, so use that mode directly here.
        run([sys.executable, source / "scripts/bitcodes/compile.py", "--no-amd", "--nvidia"],
            source / "scripts/bitcodes", env)
    exported = build / "install"
    run(["cmake", "--install", build / "cmake", "--config", "Release", "--prefix", exported], root, env)
    headers = exported / "include/hiprt"
    header = require_file(headers / "hiprt.h").read_text(encoding="utf-8")
    match = re.search(r'^\s*#define\s+HIPRT_VERSION_STR\s+"(\d+)"', header, re.MULTILINE)
    if not match:
        raise RuntimeError("Cannot read HIPRT_VERSION_STR from the installed hiprt/hiprt.h")
    prefix = "hiprt" + match.group(1)
    output = exported / "bin"
    artifacts = [require_file(output / (prefix + "64.dll" if WINDOWS else "lib" + prefix + "64.so"))]
    if WINDOWS:
        artifacts.append(require_file(output / (prefix + "64.lib")))
    if use_amd:
        hip_version = re.search(r'^\s*#define\s+HIP_VERSION_STR\s+"([^\"]+)"', header, re.MULTILINE)
        if not hip_version:
            raise RuntimeError("Cannot read HIP_VERSION_STR from the installed hiprt/hiprt.h")
        for name in (prefix + "_" + hip_version.group(1) + "_amd_lib_" + ("win" if WINDOWS else "linux") + ".bc",
                     prefix + "_" + hip_version.group(1) + "_amd.hipfb", "oro_compiled_kernels.hipfb"):
            artifacts.append(require_file(output / name))
    if use_cuda:
        # HIPRT's install rules currently export the AMD kernels only.
        # Its official compile.py exports the CUDA kernels to dist/bin/Release.
        for name in (prefix + "_nv_lib.fatbin", prefix + "_nv.fatbin", "oro_compiled_kernels.fatbin"):
            artifacts.append(require_file(source / "dist/bin/Release" / name))
    metadata = {"repository": SOURCE_URL, "ref": args.ref, "revision": revision,
                "version": match.group(1), "library": prefix + "64", "backend": args.backend,
                "kernel_build": "HIPRT CMake PRECOMPILE" if use_amd else "HIPRT compile.py --no-amd --nvidia",
                "hip_path": str(hip) if hip else None, "python": sys.version,
                "artifacts": [item.name for item in artifacts]}
    install(root, build, headers, artifacts, hip, use_amd, metadata)
    print("HIP RT is ready. Generate the tutorial projects with Premake (see README.md).", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, subprocess.CalledProcessError) as error:
        print("ERROR: " + str(error), file=sys.stderr)
        sys.exit(1)
