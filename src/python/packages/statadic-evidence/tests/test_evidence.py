"""Evidence inventories and archives include hidden and nested evidence."""

import io
import tarfile

import pytest
from statadic_evidence import (
    extract_archive,
    fingerprint,
    inventory,
    sha256,
    verify_inventory,
    write_archive,
    write_json,
)


def test_canonical_atomic_json(tmp_path):
    path = tmp_path / "data.json"
    write_json(path, {"z": "math\nα", "a": 1})
    first = path.read_bytes()
    write_json(path, {"a": 1, "z": "math\nα"})
    assert path.read_bytes() == first
    assert fingerprint({"a": 1, "b": 2}) == fingerprint({"b": 2, "a": 1})
    with pytest.raises(ValueError):
        write_json(path, {"value": float("nan")})
    assert path.read_bytes() == first
    assert list(tmp_path.iterdir()) == [path]


def test_exact_exclusions_and_hidden_inventory(tmp_path):
    write_json(tmp_path / "manifest.json", {})
    write_json(tmp_path / "nested/manifest.json", {"retained": True})
    (tmp_path / ".buildinfo").write_text("hidden")
    expected = inventory(tmp_path)
    assert set(expected) == {"nested/manifest.json", ".buildinfo"}
    verify_inventory(tmp_path, expected)
    (tmp_path / "unlisted").write_text("extra")
    with pytest.raises(ValueError, match="inventory"):
        verify_inventory(tmp_path, expected)
    (tmp_path / "unlisted").unlink()
    (tmp_path / ".buildinfo").unlink()
    with pytest.raises(ValueError, match="inventory"):
        verify_inventory(tmp_path, expected)


@pytest.mark.parametrize(
    "unsafe", ["../escape", "/absolute", "a/../b", "a\\b", "a//b", "./a"]
)
def test_unsafe_inventory_paths(tmp_path, unsafe):
    with pytest.raises(ValueError, match="Unsafe"):
        inventory(tmp_path, exclude=(unsafe,))
    with pytest.raises(ValueError, match="Unsafe"):
        verify_inventory(tmp_path, {unsafe: "0" * 64})


def test_symlinks_rejected(tmp_path):
    (tmp_path / "link").symlink_to("missing")
    with pytest.raises(ValueError, match="Unsupported"):
        inventory(tmp_path)
    with pytest.raises(ValueError):
        sha256(tmp_path / "link")
    with pytest.raises(ValueError):
        write_json(tmp_path / "link", {})


def test_archive_roundtrip_is_deterministic(tmp_path):
    root = tmp_path / "evidence"
    write_json(root / "manifest.json", {"status": "complete"})
    write_json(root / "nested/manifest.json", {"proof": "linked"})
    (root / ".buildinfo").write_text("hidden")
    (root / "empty").mkdir()
    first, second = tmp_path / "first.tar", tmp_path / "second.tar"
    write_archive(root, first)
    (root / ".buildinfo").touch()
    write_archive(root, second)
    assert first.read_bytes() == second.read_bytes()
    restored = tmp_path / "restored"
    extract_archive(first, restored)
    assert inventory(restored, exclude=()) == inventory(root, exclude=())
    assert (restored / "empty").is_dir()
    with pytest.raises(ValueError, match="outside"):
        write_archive(root, root / "self.tar")
    with pytest.raises(ValueError, match="must not exist"):
        extract_archive(first, restored)


@pytest.mark.parametrize(
    "kind", ["traversal", "symlink", "hardlink", "duplicate", "parent-file"]
)
def test_malicious_archives_rejected_before_writing(tmp_path, kind):
    source = tmp_path / "bad.tar"
    with tarfile.open(source, "w") as archive:
        info = tarfile.TarInfo("../escape" if kind == "traversal" else "entry")
        if kind in {"symlink", "hardlink"}:
            info.type = tarfile.SYMTYPE if kind == "symlink" else tarfile.LNKTYPE
            info.linkname = "../outside"
        archive.addfile(info, io.BytesIO())
        if kind == "duplicate":
            archive.addfile(info, io.BytesIO())
        if kind == "parent-file":
            archive.addfile(tarfile.TarInfo("entry/child"), io.BytesIO())
    destination = tmp_path / "restored"
    with pytest.raises(ValueError):
        extract_archive(source, destination)
    assert not destination.exists()
