"""Checkpoint download and location management for VeinSeg."""
import os
from pathlib import Path

HF_REPO     = "YousifKhoury/VeinSeg"
HF_FILENAME = "checkpoint.pth"

# Where the installed checkpoint path is recorded
_CONFIG_DIR  = Path.home() / ".config" / "veinseg"
_CONFIG_FILE = _CONFIG_DIR / "checkpoint_path"


def get_checkpoint() -> str:
    """Return the checkpoint path, checking env var then config file."""

    # Allow override via environment variable
    env = os.environ.get("VEINSEG_CHECKPOINT")
    if env:
        if not Path(env).exists():
            raise FileNotFoundError(
                f"VEINSEG_CHECKPOINT is set to '{env}' but the file does not exist."
            )
        return env

    # Read from config written by veinseg-install
    if _CONFIG_FILE.exists():
        path = _CONFIG_FILE.read_text().strip()
        if Path(path).exists():
            return path
        raise FileNotFoundError(
            f"Checkpoint path recorded in {_CONFIG_FILE} no longer exists:\n"
            f"  {path}\n"
            f"Re-run: veinseg-install <dir>"
        )

    raise RuntimeError(
        "No checkpoint found. Run first:\n"
        "  veinseg-install /path/to/dir\n"
        "Or set the VEINSEG_CHECKPOINT environment variable."
    )


def install_main():
    """Entry point for `veinseg-install <dir>`."""
    import sys
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("usage: veinseg-install <directory>")
        print()
        print("  Downloads the VeinSeg checkpoint (~600 MB) from Hugging Face")
        print("  into <directory>/checkpoint.pth and records the path so that")
        print("  the veinseg command can find it automatically.")
        print()
        print("  The path can also be overridden at any time with:")
        print("    export VEINSEG_CHECKPOINT=/path/to/checkpoint.pth")
        sys.exit(0)

    install_dir = Path(sys.argv[1]).expanduser().resolve()
    install_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = install_dir / "checkpoint.pth"

    if ckpt_path.exists():
        print(f"[veinseg-install] checkpoint already exists at {ckpt_path}")
    else:
        print(f"[veinseg-install] downloading {HF_REPO}/{HF_FILENAME} ...")
        print( "[veinseg-install] (~600 MB, this only happens once)")
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            repo_id=HF_REPO,
            filename=HF_FILENAME,
            local_dir=str(install_dir),
        )
        print(f"[veinseg-install] downloaded to {ckpt_path}")

    # Record the path
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    _CONFIG_FILE.write_text(str(ckpt_path))

    print()
    print(f"  checkpoint : {ckpt_path}")
    print(f"  config     : {_CONFIG_FILE}")
    print()
    print("  Ready. Run:")
    print("    veinseg -i qsm.nii.gz -r tgv -f 7t -o mask.nii.gz -p prob.nii.gz")
