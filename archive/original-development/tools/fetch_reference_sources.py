#!/usr/bin/env python3
"""Restore large, pinned upstream source trees excluded from the BRAIDS repo."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "delft3d": (
        ROOT / "resources/Delft3D-main",
        "https://github.com/Deltares/Delft3D.git",
        "88e9750548eddd2fd7c6ca27395f9199bf148bd5",
    ),
    "geoflood": (
        ROOT / "resources/GeoFlood-master",
        "https://github.com/KYANJO/GeoFlood.git",
        "cf9a82345e77c614c2ccd93e57cf392af687b763",
    ),
    "clawpack-tutorial": (
        ROOT / "resources/Clawpack_tutorial",
        "https://github.com/cancey/Clawpack_tutorial.git",
        "ad3e22312d2370a324de977f7bfa4ab76e85b30e",
    ),
    "avac": (
        ROOT / "vendor/avac-qgis",
        "https://github.com/cgotelli/AVAC-QGIS.git",
        "cfbaaf4d2c6a8c4bbb648dcfb5749fe488b30b32",
    ),
    "hydrosedfoam": (
        ROOT / "validation/vendor/HydroSedFoam",
        "https://github.com/ZhenduoZhu/HydroSedFoam.git",
        "f1390a407e66d7cc31bb3e93d2b0787738a17454",
    ),
}


def fetch(name: str, *, replace: bool) -> None:
    destination, url, revision = SOURCES[name]
    if destination.exists():
        if not replace:
            print(f"{name}: already present at {destination}")
            return
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", url,
                    str(destination)], check=True)
    subprocess.run(["git", "-C", str(destination), "checkout", "--detach", revision],
                   check=True)
    actual = subprocess.check_output(
        ["git", "-C", str(destination), "rev-parse", "HEAD"], text=True).strip()
    if actual != revision:
        raise RuntimeError(f"{name}: expected {revision}, checked out {actual}")
    print(f"{name}: {revision}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", choices=sorted(SOURCES),
                        help="sources to restore; omitted means all")
    parser.add_argument("--replace", action="store_true",
                        help="replace an existing working copy")
    args = parser.parse_args()
    for name in args.names or SOURCES:
        fetch(name, replace=args.replace)


if __name__ == "__main__":
    main()
