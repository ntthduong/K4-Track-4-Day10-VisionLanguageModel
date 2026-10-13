"""Resolve Colab's optional torchao conflict before constructing PEFT adapters."""
import importlib.metadata
import subprocess
import sys


def prepare():
    from packaging.version import Version
    try:
        version = importlib.metadata.version("torchao")
    except importlib.metadata.PackageNotFoundError:
        return
    if Version(version) < Version("0.16.0"):
        # NF4 in this lab uses bitsandbytes, not torchao. Avoid replacing Colab's torch.
        print(f"Removing unused torchao {version}: incompatible with PEFT 0.21.2", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=True)


if __name__ == "__main__":
    prepare()
