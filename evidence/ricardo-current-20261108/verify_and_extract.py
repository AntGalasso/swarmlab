"""Verify and optionally extract the inert research evidence archives."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
import zipfile


def safe_path(root, name):
    target = (root / name).resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"Unsafe archive path: {name}")
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "MANIFEST.json").read_text())
    verified = []
    for archive in manifest["archives"]:
        chunks = []
        for part in archive["parts"]:
            data = safe_path(root, part["path"]).read_bytes()
            if len(data) != part["bytes"]:
                raise ValueError(f"Part size mismatch: {part['path']}")
            if hashlib.sha256(data).hexdigest() != part["sha256"]:
                raise ValueError(f"Part hash mismatch: {part['path']}")
            chunks.append(data)
        data = b"".join(chunks)
        if len(data) != archive["bytes"]:
            raise ValueError(f"Archive size mismatch: {archive['name']}")
        if hashlib.sha256(data).hexdigest() != archive["sha256"]:
            raise ValueError(f"Archive hash mismatch: {archive['name']}")
        verified.append((archive, data))
        print(f"PASS {archive['name']}: {len(data)} bytes")
    if args.extract:
        output = args.extract.resolve()
        if output.exists():
            raise FileExistsError("Choose a new output directory; existing files are never overwritten")
        output.mkdir(parents=True)
        for archive, data in verified:
            name = archive["name"]
            (output / name).write_bytes(data)
            folder = output / name.removesuffix(".tar.gz").removesuffix(".zip")
            folder.mkdir()
            if name.endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(data)) as handle:
                    for member in handle.infolist():
                        target = safe_path(folder, member.filename)
                        if (member.external_attr >> 16) & 0o170000 == 0o120000:
                            raise ValueError("Symbolic links are not extracted")
                        if member.is_dir():
                            target.mkdir(parents=True, exist_ok=True)
                        else:
                            target.parent.mkdir(parents=True, exist_ok=True)
                            target.write_bytes(handle.read(member))
            else:
                with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as handle:
                    for member in handle:
                        target = safe_path(folder, member.name)
                        if member.isdir():
                            target.mkdir(parents=True, exist_ok=True)
                        elif member.isfile():
                            target.parent.mkdir(parents=True, exist_ok=True)
                            target.write_bytes(handle.extractfile(member).read())
                        else:
                            raise ValueError("Unsupported archive member")
        objects = output / "source-objects" / "objects"
        for item in objects.iterdir():
            data = item.read_bytes()
            digest = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
            if digest != item.name:
                raise ValueError(f"Source object hash mismatch: {item.name}")
        print(f"PASS {len(list(objects.iterdir()))} exact source Git blobs")
    print("Verification does not execute experiments or establish scientific validity.")


if __name__ == "__main__":
    main()
