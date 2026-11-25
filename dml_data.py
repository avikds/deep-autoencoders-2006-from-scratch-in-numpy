"""Small local MNIST loader compatible with this project's scaffold.py.

The module provides ``load('mnist')`` and returns an object with
``x_train``, ``y_train``, ``x_test`` and ``y_test`` attributes.

On the first call, the four official MNIST IDX gzip archives are downloaded
into ``data/mnist`` next to this file. Later calls reuse the cached archives.
Only NumPy and Python's standard library are required.
"""

from __future__ import annotations

import gzip
import hashlib
import os
import struct
import tempfile
from pathlib import Path
from types import SimpleNamespace
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np


# Official MNIST archive names and MD5 checksums. Checksums let us reject
# incomplete downloads and files that have been corrupted in the local cache.
_ARCHIVES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}

# Public mirrors of the same original MNIST IDX gzip files.
_BASE_URLS = (
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://azureopendatastorage.blob.core.windows.net/mnist/",
)

_DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data" / "mnist"
_CACHE = {}


def _md5(path: Path) -> str:
    """Return the MD5 checksum of a file, reading it in bounded-size blocks."""
    digest = hashlib.md5()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_archive(filename: str, destination: Path) -> None:
    """Download one verified MNIST archive, trying each configured mirror."""
    expected_md5 = _ARCHIVES[filename]

    # Reuse a valid cached archive without making a network request.
    if destination.is_file() and _md5(destination) == expected_md5:
        return

    # Remove an invalid cache entry before attempting a clean download.
    if destination.exists():
        destination.unlink()

    errors = []
    for base_url in _BASE_URLS:
        url = base_url + filename
        temp_path = None
        try:
            request = Request(url, headers={"User-Agent": "dml-data-mnist-loader/1.0"})
            with urlopen(request, timeout=60) as response:
                fd, temp_name = tempfile.mkstemp(
                    prefix=filename + ".", suffix=".part", dir=destination.parent
                )
                temp_path = Path(temp_name)
                digest = hashlib.md5()
                with os.fdopen(fd, "wb") as output:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)
                        digest.update(chunk)
                    output.flush()
                    os.fsync(output.fileno())

            actual_md5 = digest.hexdigest()
            if actual_md5 != expected_md5:
                raise ValueError(
                    f"Checksum mismatch for {filename}: expected {expected_md5}, "
                    f"received {actual_md5}"
                )

            # Atomic rename prevents a partial download from looking complete.
            temp_path.replace(destination)
            return

        except Exception as exc:  # Try the next mirror, retaining the reason.
            errors.append(f"{url}: {exc}")
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass

    details = "\n  ".join(errors)
    raise RuntimeError(
        f"Could not download {filename} from the MNIST mirrors. "
        "Check your internet connection or download the four MNIST IDX gzip "
        f"archives into {destination.parent}. Tried:\n  {details}"
    )


def _read_images(path: Path) -> np.ndarray:
    """Read an IDX image archive as uint8 with shape (n, rows, columns)."""
    with gzip.open(path, "rb") as file:
        header = file.read(16)
        if len(header) != 16:
            raise ValueError(f"Invalid MNIST image header in {path}")
        magic, count, rows, cols = struct.unpack(">IIII", header)
        if magic != 2051:
            raise ValueError(f"Invalid MNIST image magic number in {path}: {magic}")
        payload = file.read()

    expected_size = count * rows * cols
    if len(payload) != expected_size:
        raise ValueError(
            f"Invalid image payload in {path}: expected {expected_size} bytes, "
            f"found {len(payload)}"
        )
    if (rows, cols) != (28, 28):
        raise ValueError(f"Expected 28x28 MNIST images, found {rows}x{cols}")

    return np.frombuffer(payload, dtype=np.uint8).reshape(count, rows, cols).copy()


def _read_labels(path: Path) -> np.ndarray:
    """Read an IDX label archive as uint8 with shape (n,)."""
    with gzip.open(path, "rb") as file:
        header = file.read(8)
        if len(header) != 8:
            raise ValueError(f"Invalid MNIST label header in {path}")
        magic, count = struct.unpack(">II", header)
        if magic != 2049:
            raise ValueError(f"Invalid MNIST label magic number in {path}: {magic}")
        payload = file.read()

    if len(payload) != count:
        raise ValueError(
            f"Invalid label payload in {path}: expected {count} bytes, "
            f"found {len(payload)}"
        )

    labels = np.frombuffer(payload, dtype=np.uint8).copy()
    if np.any(labels > 9):
        raise ValueError(f"MNIST labels in {path} must be digits 0 through 9")
    return labels


def load(name: str = "mnist", data_dir=None, download: bool = True):
    """Load MNIST and return train/test arrays in a scaffold-compatible object.

    Parameters
    ----------
    name:
        Dataset name. This project uses ``"mnist"``.
    data_dir:
        Optional directory for the four compressed IDX files. By default,
        files are stored in ``data/mnist`` next to ``dml_data.py``.
    download:
        If True (default), download archives that are not already cached.
        If False, raise FileNotFoundError when any archive is missing.

    Returns
    -------
    types.SimpleNamespace
        ``x_train`` and ``x_test`` are uint8 arrays shaped (N, 28, 28);
        ``y_train`` and ``y_test`` are uint8 arrays shaped (N,).
        Pixel values remain in [0, 255]; scaffold.py performs normalization.
    """
    if not isinstance(name, str) or name.lower() != "mnist":
        raise ValueError(f"Unsupported dataset {name!r}; this module supports only 'mnist'.")

    root = Path(data_dir) if data_dir is not None else _DEFAULT_DATA_DIR
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    cache_key = str(root)
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    paths = {filename: root / filename for filename in _ARCHIVES}
    for filename, path in paths.items():
        if path.is_file() and _md5(path) == _ARCHIVES[filename]:
            continue
        if not download:
            raise FileNotFoundError(
                f"Missing or invalid MNIST archive: {path}. "
                "Call load('mnist', download=True) to download it."
            )
        _download_archive(filename, path)

    x_train = _read_images(paths["train-images-idx3-ubyte.gz"])
    y_train = _read_labels(paths["train-labels-idx1-ubyte.gz"])
    x_test = _read_images(paths["t10k-images-idx3-ubyte.gz"])
    y_test = _read_labels(paths["t10k-labels-idx1-ubyte.gz"])

    if len(x_train) != len(y_train) or len(x_test) != len(y_test):
        raise ValueError("MNIST image and label counts do not match.")
    if (len(x_train), len(x_test)) != (60000, 10000):
        raise ValueError(
            "Unexpected MNIST split sizes: expected 60,000 training and 10,000 "
            f"test examples, found {len(x_train)} and {len(x_test)}."
        )

    dataset = SimpleNamespace(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
    )
    _CACHE[cache_key] = dataset
    return dataset


__all__ = ["load"]
