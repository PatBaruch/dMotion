"""Install the built wheel outside the source tree and check the shipped CLI."""

import argparse
import os
import subprocess
import tempfile
import tomllib
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uv", default="uv")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    wheels = list((root / "dist").glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError("Build exactly one wheel in a clean dist directory first")
    version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    env = {
        key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "VIRTUAL_ENV"}
    }
    with tempfile.TemporaryDirectory(prefix="dmotion-package-") as directory:
        temporary = Path(directory)
        environment = temporary / "venv"
        requirements = temporary / "runtime.txt"
        subprocess.run(
            [
                args.uv,
                "export",
                "--locked",
                "--no-dev",
                "--no-emit-project",
                "-o",
                str(requirements),
            ],
            cwd=root,
            env=env,
            check=True,
        )
        subprocess.run([args.uv, "venv", str(environment)], env=env, check=True)
        python = environment / "bin" / "python"
        subprocess.run(
            [
                args.uv,
                "pip",
                "install",
                "--python",
                str(python),
                "--require-hashes",
                "-r",
                str(requirements),
            ],
            env=env,
            check=True,
        )
        subprocess.run(
            [args.uv, "pip", "install", "--python", str(python), "--no-deps", str(wheels[0])],
            env=env,
            check=True,
        )
        command = environment / "bin" / "dmotion"
        result = subprocess.run(
            [str(command), "--version"],
            cwd=temporary,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        if result.stdout.strip() != version:
            raise ValueError("Installed CLI version does not match package metadata")
        subprocess.run([str(command), "--help"], cwd=temporary, env=env, check=True)
        subprocess.run(
            [
                str(python),
                "-c",
                "import importlib.util; from dmotion.cli import parser; parser(); "
                "assert all(importlib.util.find_spec(m) is None for m in "
                "('torch', 'cv2', 'transformers', 'pygame')), 'Unexpected vision dependencies'",
            ],
            cwd=temporary,
            env=env,
            check=True,
        )
    print(f"Wheel {version} installs and runs outside the repository without vision dependencies.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
