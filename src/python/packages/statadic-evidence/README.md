# Evidence files

Internal standard-library-only workspace package for exact inventories and safe
portable archives. It does not judge scientific validity. Run its tests with
`just python::evidence::check` in the pinned environment.

`write_json` writes canonical finite UTF-8 JSON atomically. `fingerprint` hashes
those same bytes; `sha256` hashes regular files. `inventory` includes hidden files
and excludes only exact root-relative names (by default the root `manifest.json`).
Nested manifests remain evidence. `verify_inventory` rejects changed, missing and
unlisted files and unsafe paths. Inventories and archives reject symbolic links.

After verifying its own scientific completeness and inventory, a caller may use
`write_archive(root, destination)` to retain deterministic uncompressed tar bytes.
Root manifests and empty directories are included. `extract_archive` validates
all members before writing into a new destination, rejecting traversal, links,
special files, duplicates and file/directory collisions. Restored evidence must
then pass its caller's verifier.
