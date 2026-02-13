"""
Manager classes for llama.cpp and HuggingFace operations.
"""
import os
import shutil
import asyncio
import logging
import platform
import subprocess
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import HfApi

logger = logging.getLogger("GGUF_Forge")

# These will be set by main app
LLAMA_CPP_DIR = None
BASE_DIR = None

# Thread pool for blocking operations
_executor = ThreadPoolExecutor(max_workers=4)


def set_paths(base_dir: Path, llama_cpp_dir: Path):
    """Set the paths for managers."""
    global BASE_DIR, LLAMA_CPP_DIR
    BASE_DIR = base_dir
    LLAMA_CPP_DIR = llama_cpp_dir


class LlamaCppManager:
    @staticmethod
    def is_installed() -> bool:
        return (LLAMA_CPP_DIR / "CMakeLists.txt").exists()
    
    @staticmethod
    def check_tool(tool_name: str) -> bool:
        """Check if a tool is available in PATH."""
        return shutil.which(tool_name) is not None
    
    @staticmethod
    async def has_nvidia_gpu() -> bool:
        """Check if NVIDIA GPU is available (async)."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "nvidia-smi",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )
            returncode = await proc.wait()
            return returncode == 0
        except FileNotFoundError:
            return False
    
    @staticmethod
    async def clone_repo(force: bool = False):
        """Clone or update llama.cpp repository.
        
        Args:
            force: If True, forcefully fetch and reset to latest remote commit,
                   discarding any local changes. If False, perform normal git pull.
        """
        if LlamaCppManager.is_installed():
            if force:
                logger.info("llama.cpp already exists. Forcefully updating to latest commit...")
                
                # Step 1: Fetch all updates from origin
                logger.info("Fetching all updates from origin...")
                proc = await asyncio.create_subprocess_exec(
                    "git", "fetch", "--all",
                    cwd=LLAMA_CPP_DIR,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await proc.communicate()
                if proc.returncode != 0:
                    error_msg = stderr.decode() if stderr else "Unknown error"
                    logger.error(f"Failed to fetch updates: {error_msg}")
                    raise Exception(f"Failed to fetch llama.cpp updates: {error_msg}")
                
                # Step 2: Get current branch name
                proc = await asyncio.create_subprocess_exec(
                    "git", "rev-parse", "--abbrev-ref", "HEAD",
                    cwd=LLAMA_CPP_DIR,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await proc.communicate()
                if proc.returncode != 0:
                    # Fallback to master/main if branch detection fails
                    branch = "master"
                    logger.warning(f"Could not detect branch, defaulting to {branch}")
                else:
                    branch = stdout.decode().strip()
                    logger.info(f"Current branch: {branch}")
                
                # Step 3: Hard reset to origin branch
                logger.info(f"Resetting to origin/{branch} (discarding local changes)...")
                proc = await asyncio.create_subprocess_exec(
                    "git", "reset", "--hard", f"origin/{branch}",
                    cwd=LLAMA_CPP_DIR,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await proc.communicate()
                if proc.returncode != 0:
                    error_msg = stderr.decode() if stderr else "Unknown error"
                    logger.error(f"Failed to reset to origin/{branch}: {error_msg}")
                    raise Exception(f"Failed to reset llama.cpp: {error_msg}")
                
                # Step 4: Get current commit info
                proc = await asyncio.create_subprocess_exec(
                    "git", "log", "-1", "--oneline",
                    cwd=LLAMA_CPP_DIR,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await proc.communicate()
                if proc.returncode == 0:
                    commit_info = stdout.decode().strip()
                    logger.info(f"Updated to: {commit_info}")
                else:
                    logger.info("Force update complete")
            else:
                logger.info("llama.cpp already exists. Pulling latest...")
                proc = await asyncio.create_subprocess_exec(
                    "git", "pull",
                    cwd=LLAMA_CPP_DIR,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await proc.wait()
        else:
            logger.info("Cloning llama.cpp...")
            proc = await asyncio.create_subprocess_exec(
                "git", "clone", "https://github.com/ggerganov/llama.cpp", str(LLAMA_CPP_DIR),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            returncode = await proc.wait()
            if returncode != 0:
                raise Exception("Failed to clone llama.cpp")

    @staticmethod
    async def build():
        """Build llama.cpp using CMake with optional CUDA support."""
        logger.info("Building llama.cpp...")
        system = platform.system()

        # Check if already built - skip if llama-quantize exists
        try:
            existing = LlamaCppManager.get_quantize_path()
            if existing.exists():
                logger.info(f"llama.cpp already built, skipping. Found: {existing}")
                return
        except FileNotFoundError:
            pass  # Not built yet, continue with build

        # Check if cmake is available
        if not LlamaCppManager.check_tool("cmake"):
            raise Exception("CMake is not installed or not in PATH. Please install CMake.")

        # Check for CUDA support
        has_cuda = await LlamaCppManager.has_nvidia_gpu()
        if has_cuda:
            logger.info("NVIDIA GPU detected, building with CUDA support...")
        else:
            logger.info("No NVIDIA GPU detected, building CPU-only version...")

        build_dir = LLAMA_CPP_DIR / "build"

        # Clean build directory if it exists but might be corrupted
        if build_dir.exists():
            cmake_cache = build_dir / "CMakeCache.txt"
            if cmake_cache.exists():
                logger.info("Existing build directory found, cleaning...")
                shutil.rmtree(build_dir, ignore_errors=True)

        build_dir.mkdir(exist_ok=True)

        try:
            # Step 1: CMake Configure
            cmake_args = [
                "cmake", "..",
                "-DGGML_CUDA=OFF" if not has_cuda else "-DGGML_CUDA=ON",
                "-DCMAKE_BUILD_TYPE=Release",
                "-DGGML_NATIVE=OFF",  # Disable native optimizations for broader compatibility
            ]

            # Platform-specific CMake generator and settings
            if system == "Windows":
                # Try different generators in order of preference
                generators = [
                    ("Visual Studio 17 2022", "x64"),
                    ("Visual Studio 16 2019", "x64"),
                    ("MinGW Makefiles", None),
                    ("NMake Makefiles", None),
                ]

                cmake_success = False
                last_error = ""

                for generator, arch in generators:
                    logger.info(f"Trying CMake generator: {generator}")
                    test_args = cmake_args.copy()
                    test_args.extend(["-G", generator])
                    if arch:
                        test_args.extend(["-A", arch])

                    proc = await asyncio.create_subprocess_exec(
                        *test_args,
                        cwd=build_dir,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.STDOUT,
                        env={**os.environ, "CMAKE_GENERATOR": generator}
                    )
                    stdout, _ = await proc.communicate()

                    if proc.returncode == 0:
                        cmake_success = True
                        logger.info(f"CMake configure successful with {generator}")
                        break
                    else:
                        last_error = stdout.decode() if stdout else "No output"
                        logger.warning(f"CMake failed with {generator}: {last_error[:500]}")
                        # Clean build dir for next attempt
                        if build_dir.exists():
                            shutil.rmtree(build_dir, ignore_errors=True)
                        build_dir.mkdir(exist_ok=True)

                if not cmake_success:
                    raise Exception(f"CMake configure failed with all generators. Last error:\n{last_error[:2000]}")
            else:
                # Linux/macOS - simpler approach
                cmake_args.extend(["-DGGML_BLAS=OFF"])  # Disable BLAS for simpler builds

                logger.info(f"Running CMake configure: {' '.join(cmake_args)}")

                proc = await asyncio.create_subprocess_exec(
                    *cmake_args,
                    cwd=build_dir,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT
                )
                stdout, _ = await proc.communicate()

                if proc.returncode != 0:
                    error_output = stdout.decode() if stdout else "No output"
                    logger.error(f"CMake configure failed:\n{error_output}")
                    raise Exception(f"CMake configure failed. Output:\n{error_output[:2000]}")

                logger.info("CMake configure successful")

            # Step 2: CMake Build
            build_args = ["cmake", "--build", ".", "--config", "Release"]

            # Use multiple cores for faster builds
            import multiprocessing
            cores = multiprocessing.cpu_count()

            if system == "Windows":
                build_args.extend(["--", "/m"])  # Parallel build for MSBuild
            else:
                build_args.extend(["-j", str(cores)])

            logger.info(f"Running CMake build: {' '.join(build_args)}")

            proc = await asyncio.create_subprocess_exec(
                *build_args,
                cwd=build_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT
            )
            stdout, _ = await proc.communicate()

            if proc.returncode != 0:
                error_output = stdout.decode() if stdout else "No output"
                logger.error(f"CMake build failed:\n{error_output}")
                raise Exception(f"CMake build failed. Output:\n{error_output[:2000]}")

            cuda_status = "with CUDA" if has_cuda else "CPU-only"
            logger.info(f"Build successful ({cuda_status}) on {system}")

        except Exception as e:
            logger.error(f"Build failed: {e}")
            raise Exception(
                f"Failed to build llama.cpp: {str(e)}\n\n"
                "Ensure build tools are installed:\n"
                "  Windows:\n"
                "    - Visual Studio Build Tools 2019 or 2022\n"
                "    - CMake (https://cmake.org/download/)\n"
                "    - Or use: winget install Microsoft.VisualStudio.2022.BuildTools\n"
                "    - Or use: winget install Kitware.CMake\n\n"
                "  Linux (Ubuntu/Debian):\n"
                "    - sudo apt install build-essential cmake\n"
                "    - For CUDA: sudo apt install nvidia-cuda-toolkit\n\n"
                "  macOS:\n"
                "    - xcode-select --install\n"
                "    - brew install cmake"
            )

    @staticmethod
    def get_quantize_path() -> Path:
        """Find the llama-quantize executable."""
        system = platform.system()
        build_dir = LLAMA_CPP_DIR / "build"
        
        # Common paths based on CMake output
        if system == "Windows":
            candidates = [
                build_dir / "bin" / "Release" / "llama-quantize.exe",
                build_dir / "Release" / "llama-quantize.exe",
                build_dir / "bin" / "llama-quantize.exe",
                LLAMA_CPP_DIR / "build" / "llama-quantize.exe",
            ]
        else:
            candidates = [
                build_dir / "bin" / "llama-quantize",
                build_dir / "llama-quantize",
                LLAMA_CPP_DIR / "llama-quantize",
            ]
        
        for path in candidates:
            if path.exists():
                logger.info(f"Found llama-quantize at: {path}")
                return path
        
        # Fallback: recursive search
        pattern = "llama-quantize.exe" if system == "Windows" else "llama-quantize"
        found = list(LLAMA_CPP_DIR.rglob(pattern))
        if found:
            # Prefer executables in build directories
            for f in found:
                if "build" in str(f):
                    logger.info(f"Found llama-quantize at: {f}")
                    return f
            logger.info(f"Found llama-quantize at: {found[0]}")
            return found[0]
        
        raise FileNotFoundError(
            "llama-quantize executable not found. Build might have failed.\n"
            f"Searched in: {LLAMA_CPP_DIR}"
        )

    @staticmethod
    def get_gguf_split_path() -> Path:
        """Find the llama-gguf-split executable."""
        system = platform.system()
        build_dir = LLAMA_CPP_DIR / "build"
        
        if system == "Windows":
            candidates = [
                build_dir / "bin" / "Release" / "llama-gguf-split.exe",
                build_dir / "Release" / "llama-gguf-split.exe",
                build_dir / "bin" / "llama-gguf-split.exe",
                build_dir / "llama-gguf-split.exe",
                build_dir / "bin" / "Release" / "gguf-split.exe",
                build_dir / "Release" / "gguf-split.exe",
                build_dir / "bin" / "gguf-split.exe",
                build_dir / "gguf-split.exe",
            ]
        else:
            candidates = [
                build_dir / "bin" / "llama-gguf-split",
                build_dir / "llama-gguf-split",
                LLAMA_CPP_DIR / "llama-gguf-split",
                build_dir / "bin" / "gguf-split",
                build_dir / "gguf-split",
                LLAMA_CPP_DIR / "gguf-split",
            ]
        
        for path in candidates:
            if path.exists():
                logger.info(f"Found llama-gguf-split at: {path}")
                return path
        
        patterns = ["llama-gguf-split.exe", "gguf-split.exe"] if system == "Windows" else ["llama-gguf-split", "gguf-split"]
        for pattern in patterns:
            found = list(LLAMA_CPP_DIR.rglob(pattern))
            if found:
                for f in found:
                    if "build" in str(f):
                        logger.info(f"Found llama-gguf-split at: {f}")
                        return f
                logger.info(f"Found llama-gguf-split at: {found[0]}")
                return found[0]
        
        raise FileNotFoundError(
            "llama-gguf-split executable not found. Build might have failed.\n"
            f"Searched in: {LLAMA_CPP_DIR}"
        )


class HuggingFaceManager:
    def __init__(self, token: Optional[str] = None):
        self.api = HfApi(token=token)

    async def search_models(self, query: str, limit: int = 10):
        """Search for models on HuggingFace (async)."""
        loop = asyncio.get_event_loop()
        models = await loop.run_in_executor(
            _executor,
            lambda: list(self.api.list_models(search=query, limit=limit, sort="likes", direction=-1))
        )
        return [{"id": m.modelId, "likes": m.likes} for m in models]

    async def check_exists(self, repo_id: str) -> bool:
        """Check if a model exists on HuggingFace (async)."""
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                _executor,
                lambda: self.api.model_info(repo_id)
            )
            return True
        except:
            return False


async def get_app_version() -> str:
    """Calculate app version based on git commit count * 0.1 (async)."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "git", "rev-list", "--count", "HEAD",
            cwd=BASE_DIR,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL
        )
        stdout, _ = await proc.communicate()
        if proc.returncode == 0:
            commit_count = int(stdout.decode().strip())
            version = commit_count * 0.1
            return f"v{version:.1f}"
    except Exception:
        pass
    return "v0.1"  # Fallback

