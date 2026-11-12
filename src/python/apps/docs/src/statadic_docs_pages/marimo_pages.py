"""Render executed marimo notebooks as native Sphinx pages.

A notebook ``.py`` under the docs source directory is executed once per content
hash by the public ``marimo export ipynb --include-outputs`` command in the
locked CPython environment. Markdown headings reach the table of contents and
the search index, and their ``$...$`` math reaches MathJax. Shown
code becomes a Python literal block, and outputs become images, tables and
preformatted text through the ``marimo-output`` directive. Everything from the
``## Appendix`` heading onward is dropped. A failing cell fails the build.
"""

from __future__ import annotations

import base64
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from html.parser import HTMLParser
from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path
from urllib.parse import unquote, urlsplit, urlunsplit

from docutils import nodes
from myst_parser.parsers.sphinx_ import MystParser
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.errors import ExtensionError
from sphinx.project import Project
from sphinx.util import logging
from sphinx.util.docutils import SphinxDirective

LOGGER = logging.getLogger(__name__)
APPENDIX = re.compile(r"^##\s+Appendix\s*$")
TEXT_XS = re.compile(r"^<pre class='text-xs'>(.*)</pre>$", re.DOTALL)
RAW_URL = "https://raw.githubusercontent.com/marcusinthesky/statadic/{branch}/{path}"
MOLAB_URL = (
    "https://molab.marimo.io/github/marcusinthesky/statadic/blob/{branch}/{path}"
)
MOLAB_SHIELD = "https://marimo.io/molab-shield.svg"


def is_notebook(path: Path) -> bool:
    """Tell marimo notebooks from other Python files beside the pages."""
    head = path.read_text(encoding="utf-8", errors="replace")
    return "import marimo" in head and "marimo.App(" in head


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for source in sorted(root.rglob("*.py")):
        digest.update(source.relative_to(root).as_posix().encode())
        digest.update(b"\0" + source.read_bytes() + b"\0")
    return digest.hexdigest()


def _package_digest() -> str:
    spec = find_spec("statadic")
    if spec is None or spec.origin is None:  # pragma: no cover - docs provides it
        return "none"
    return _tree_digest(Path(spec.origin).resolve().parent)


def cache_key(path: Path) -> str:
    """Hash everything that changes what executing a notebook produces."""
    digest = hashlib.sha256(path.read_bytes())
    for part in (
        os.environ.get("JAX_ENABLE_X64", ""),
        version("marimo"),
        sys.version,
        hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        hashlib.sha256(
            (Path(__file__).resolve().parents[4] / "uv.lock").read_bytes()
        ).hexdigest(),
        _package_digest(),
    ):
        digest.update(b"\0" + part.encode())
    return digest.hexdigest()[:20]


def execute(path: Path, cache: Path) -> dict:
    """Return Marimo's output-inclusive Jupyter export, cached by content."""
    cache.mkdir(parents=True, exist_ok=True)
    slug = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:8]
    entry = cache / f"{path.stem}-{slug}-{cache_key(path)}.json"
    if entry.is_file():
        return json.loads(entry.read_text())
    with tempfile.TemporaryDirectory() as scratch:
        copy = Path(scratch) / path.name
        shutil.copy2(path, copy)
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "marimo",
                "export",
                "ipynb",
                "--include-outputs",
                "--no-sandbox",
                "--sort",
                "top-down",
                str(copy),
                "-o",
                str(Path(scratch) / "executed.ipynb"),
            ],
            cwd=scratch,
            env=os.environ | {"PYTHONWARNINGS": "ignore"},
            capture_output=True,
            text=True,
            check=False,
        )
        snapshot = Path(scratch) / "executed.ipynb"
        if result.returncode != 0 or not snapshot.is_file():
            tail = (result.stdout + result.stderr)[-4000:]
            raise ExtensionError(f"marimo notebook failed: {path}\n{tail}")
        session = json.loads(snapshot.read_text())
    failed = [c["id"] for c in session["cells"] if _has_error(c)]
    if failed:
        raise ExtensionError(f"marimo notebook has failing cells {failed}: {path}")
    for stale in cache.glob(f"{path.stem}-{slug}-*.json"):
        stale.unlink()
    entry.write_text(json.dumps(session))
    return session


def _has_error(cell: dict) -> bool:
    return any(o.get("output_type") == "error" for o in cell.get("outputs", []))


def _fence(code: str, info: str) -> str:
    longest = max((len(m) for m in re.findall(r"`+", code)), default=0)
    ticks = "`" * max(3, longest + 1)
    return f"{ticks}{info}\n{code}\n{ticks}"


def _payload(items: list[tuple[str, str]]) -> str:
    return base64.b64encode(json.dumps(items).encode()).decode()


def cell_outputs(cell: dict) -> list[tuple[str, str]]:
    """Select static MIME representations from standard Jupyter outputs."""
    pairs: list[tuple[str, str]] = []
    for output in cell.get("outputs", []):
        data = output.get("data")
        if not isinstance(data, dict):
            if output.get("output_type") == "stream" and output.get("name") == "stdout":
                value = output.get("text", "")
                pairs.append(
                    ("stdout", "".join(value) if isinstance(value, list) else value)
                )
            continue
        for mime in (
            "image/svg+xml",
            "image/png",
            "text/markdown",
            "text/html",
            "text/plain",
        ):
            if mime in data:
                value = data[mime]
                pairs.append(
                    (mime, "".join(value) if isinstance(value, list) else value)
                )
                break
        else:
            pairs.extend((mime, "") for mime in data)
    return [(m, d) for m, d in pairs if d.strip() or m not in {"text/plain", "stdout"}]


def compose(notebook: dict) -> str:
    """Adapt the public Jupyter export to MyST and static output directives."""
    parts: list[str] = []
    launched = False
    for cell in notebook["cells"]:
        metadata = cell.get("metadata", {})
        marimo = metadata.get("marimo", {})
        if marimo.get("name") == "setup":
            continue
        source = cell.get("source", "")
        source = "".join(source) if isinstance(source, list) else source
        if cell["cell_type"] == "markdown":
            markdown = source
            if APPENDIX.match(markdown.strip().splitlines()[0] if markdown else ""):
                break
            if not launched and markdown.startswith("# "):
                title, _, rest = markdown.partition("\n")
                markdown = f"{title}\n\n```{{marimo-launch}}\n```\n\n{rest}"
                launched = True
            parts.append(markdown)
            continue
        hidden = (
            marimo.get("config", {}).get("hide_code", False)
            or metadata.get("jupyter", {}).get("source_hidden", False)
            or "remove-input" in metadata.get("tags", [])
        )
        if not hidden and source.strip():
            parts.append(_fence(source, "python"))
        for mime, data in cell_outputs(cell):
            if mime == "text/markdown" and "<marimo-" not in html.unescape(data):
                parts.append(data)
            else:
                if mime == "text/markdown":
                    mime, data = "text/html", html.unescape(data)
                parts.append(_fence(_payload([(mime, data)]), "{marimo-output}"))
    return "\n\n".join(parts) + "\n"


class Attributes(HTMLParser):
    """Collect the first marimo custom element and its decoded attributes."""

    def __init__(self) -> None:
        super().__init__()
        self.elements: list[tuple[str, dict[str, str]]] = []

    def handle_starttag(self, tag, attrs) -> None:
        if tag.startswith("marimo-"):
            self.elements.append((tag, {k: v or "" for k, v in attrs}))


def _text(markup: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", markup)).strip()


def _json_string(value: str) -> str:
    try:
        decoded = json.loads(value)
    except ValueError:
        return value
    return decoded if isinstance(decoded, str) else value


def _fix_math(markup: str) -> str:
    """Turn marimo's ``||(...||)`` TeX elements into MathJax delimiters."""

    def replace(match: re.Match[str]) -> str:
        body = match.group(1).replace("||(", "\\(").replace("||)", "\\)")
        body = body.replace("||[", "\\[").replace("||]", "\\]")
        return f'<span class="math notranslate nohighlight">{body}</span>'

    return re.sub(r"<marimo-tex[^>]*>(.*?)</marimo-tex>", replace, markup, flags=re.S)


def normalize_tables(markup: str) -> str:
    """Keep Marimo's native static tables and remove its private row-ID column."""

    def clean(match):
        table = match[0]
        headers = re.search(r"<thead\b[^>]*>(.*?)</thead>", table, re.S)
        if headers is None:
            return table
        columns = re.findall(r"<th\b[^>]*>.*?</th>", headers[1], re.S)
        private = next(
            (i for i, value in enumerate(columns) if _text(value) == "_marimo_row_id"),
            None,
        )
        if private is None:
            return table

        def row(match):
            cells = list(re.finditer(r"<t[hd]\b[^>]*>.*?</t[hd]>", match[0], re.S))
            if private >= len(cells):
                return match[0]
            cell = cells[private]
            return match[0][: cell.start()] + match[0][cell.end() :]

        return re.sub(r"<tr\b[^>]*>.*?</tr>", row, table, flags=re.S)

    return re.sub(r"<table\b[^>]*>.*?</table>", clean, markup, flags=re.S)


def _preformatted(text: str) -> nodes.literal_block:
    return nodes.literal_block(text, text, language="text")


def _control(text: str) -> nodes.Node:
    return nodes.paragraph(
        text=f"{text} (interactive in the notebook; shown at its default value)",
        classes=["marimo-control"],
    )


def render_html(markup: str, resolve_links=lambda value: value) -> list[nodes.Node]:
    """Degrade marimo HTML: text and tables stay, widgets become captions."""
    text = TEXT_XS.match(markup.strip())
    if text:
        return [_preformatted(html.unescape(text.group(1)))]
    scan = Attributes()
    scan.feed(markup)
    kinds = {tag: attrs for tag, attrs in scan.elements}
    if "marimo-tabs" in kinds:
        labels = json.loads(kinds["marimo-tabs"].get("data-tabs", "[]"))
        content = re.search(r"<marimo-tabs\b[^>]*>(.*?)</marimo-tabs>", markup, re.S)
        if content is None:
            raise ExtensionError("Marimo tabs export is missing its content")
        names = iter(labels)

        def heading(match):
            name = _text(next(names, "Result"))
            return (
                f'<div class="marimo-tab"><p><strong>{html.escape(name)}</strong></p>'
            )

        body = re.sub(r"<div\s+data-kind=['\"]tab['\"]>", heading, content[1])
        return [nodes.raw("", resolve_links(_fix_math(body)), format="html")]
    if "marimo-callout-output" in kinds:
        attrs = kinds["marimo-callout-output"]
        inner = _json_string(attrs.get("data-html", '""'))
        kind = _json_string(attrs.get("data-kind", '"neutral"'))
        return [
            nodes.raw(
                "",
                f'<div class="marimo-callout {html.escape(kind)}">'
                f"{resolve_links(_fix_math(inner))}</div>",
                format="html",
            )
        ]
    if "marimo-slider" in kinds:
        attrs = kinds["marimo-slider"]
        label = _text(_json_string(attrs.get("data-label", "")))
        return [
            _control(f"Slider {label}: {attrs.get('data-initial-value', '')}".strip())
        ]
    if "marimo-dropdown" in kinds:
        attrs = kinds["marimo-dropdown"]
        label = _text(_json_string(attrs.get("data-label", "")))
        value = _json_string(attrs.get("data-initial-value", ""))
        return [_control(f"Choice {label}: {value}".strip())]
    if kinds:
        tag = next(iter(kinds))
        attrs = kinds[tag]
        label = _text(_json_string(attrs.get("data-label", "")))
        return [
            _control(
                f"Interactive element {tag.removeprefix('marimo-')} {label}".strip()
            )
        ]
    return [nodes.raw("", resolve_links(normalize_tables(markup)), format="html")]


def render_output(
    mime: str, data: str, resolve_links=lambda value: value
) -> list[nodes.Node]:
    """Convert one output into docutils nodes."""
    if mime in {"image/svg+xml", "image/png"}:
        if mime == "image/svg+xml" and data.lstrip().startswith("<"):
            data = base64.b64encode(data.encode()).decode()
        uri = data if data.startswith("data:") else f"data:{mime};base64,{data}"
        return [nodes.image(uri=uri, alt="Notebook figure", classes=["marimo-figure"])]
    if mime == "text/markdown":
        return [nodes.raw("", resolve_links(_fix_math(data)), format="html")]
    if mime == "text/html":
        return render_html(data, resolve_links)
    if mime in {"text/plain", "stdout"}:
        return [_preformatted(data.rstrip("\n"))]
    return [_control(f"Output of type {mime}")]


HREF = re.compile(r"(?i)((?<![\w-])href\s*=\s*)([\"'])(.*?)\2")


def resolve_output_links(markup: str, env, builder) -> str:
    """Resolve links to Sphinx source documents inside rendered notebook HTML."""

    def replace(match: re.Match[str]) -> str:
        original = html.unescape(match.group(3))
        parts = urlsplit(original)
        if parts.scheme or parts.netloc or not parts.path:
            return match.group(0)
        if Path(unquote(parts.path)).suffix not in {".md", ".rst", ".py"}:
            return match.group(0)
        _, absolute = env.relfn2path(unquote(parts.path), env.docname)
        target = env.path2doc(absolute)
        if target is None or target not in env.found_docs:
            return match.group(0)
        uri = builder.get_relative_uri(env.docname, target)
        resolved = urlunsplit(("", "", uri, parts.query, parts.fragment))
        escaped = html.escape(resolved, quote=True)
        return f"{match.group(1)}{match.group(2)}{escaped}{match.group(2)}"

    return HREF.sub(replace, markup)


class MarimoOutput(SphinxDirective):
    """Render the outputs carried in the base64 JSON body."""

    has_content = True

    def run(self) -> list[nodes.Node]:
        items = json.loads(base64.b64decode("".join(self.content)))
        container = nodes.container(classes=["marimo-output"])
        for mime, data in items:
            container.extend(
                render_output(
                    mime,
                    data,
                    lambda markup: resolve_output_links(
                        markup, self.env, self.env.app.builder
                    ),
                )
            )
        return [container]


class MarimoLaunch(SphinxDirective):
    """Open-in-molab badge and a local run command, for notebook pages."""

    def run(self) -> list[nodes.Node]:
        config = self.config
        source = Path(self.env.doc2path(self.env.docname)).resolve()
        root = Path(config.statadic_repository_root).resolve()
        try:
            relative = source.relative_to(root).as_posix()
        except ValueError:
            relative = source.relative_to(Path(self.env.srcdir).resolve()).as_posix()
        branch = config.statadic_notebook_branch
        fields = {"branch": branch, "path": relative}
        shield = config.statadic_molab_shield
        uri = MOLAB_SHIELD
        if shield and Path(shield).is_file():
            encoded = base64.b64encode(Path(shield).read_bytes()).decode()
            uri = f"data:image/svg+xml;base64,{encoded}"
        badge = nodes.reference(
            "", "", refuri=MOLAB_URL.format(**fields), internal=False
        )
        badge += nodes.image(uri=uri, alt="Open in molab")
        command = f"uvx marimo edit --sandbox {RAW_URL.format(**fields)}"
        local = nodes.paragraph("", "Run locally: ")
        local += nodes.literal(command, command)
        download = addnodes.download_reference("", "", reftarget=source.name)
        download += nodes.literal("Download notebook", "Download notebook")
        return [
            nodes.container(
                "",
                nodes.paragraph("", "", badge),
                nodes.paragraph("", "", download),
                local,
                classes=["marimo-launch"],
            )
        ]


class MarimoParser(MystParser):
    """Parse a marimo notebook as the MyST page it renders to."""

    supported = ("marimo",)

    def parse(self, inputstring: str, document: nodes.document) -> None:
        env = document.settings.env
        path = Path(document["source"])
        cache = Path(env.config.statadic_notebook_cache)
        super().parse(compose(execute(path, cache)), document)


def exclude_other_python(app: Sphinx, config) -> None:
    """Keep non-notebook Python files, and notebook scaffolds, out of the site."""
    source = Path(app.srcdir)
    config.exclude_patterns = [
        *config.exclude_patterns,
        "**/__marimo__",
        *(
            path.relative_to(source).as_posix()
            for path in sorted(source.rglob("*.py"))
            if "__marimo__" not in path.parts and not is_notebook(path)
        ),
    ]


class ConfinedProject(Project):
    """A project whose documents all live under its source directory.

    Registering ``.py`` as a source suffix would otherwise make MyST resolve a
    link to a test module beside the package as a missing document, instead of
    a download.
    """

    def path2doc(self, filename):
        path = Path(filename)
        if path.is_absolute() and not path.is_relative_to(self.srcdir):
            return None
        return super().path2doc(filename)


def confine_documents(app: Sphinx) -> None:
    app.project.__class__ = ConfinedProject


def setup(app: Sphinx) -> dict:
    app.setup_extension("myst_parser")
    default_cache = Path(__file__).resolve().parents[2] / "build/notebook-cache"
    app.add_config_value("statadic_notebook_branch", "main", "env")
    app.add_config_value("statadic_notebook_cache", str(default_cache), "")
    app.add_config_value("statadic_repository_root", str(Path.cwd()), "env")
    app.add_config_value("statadic_molab_shield", None, "env")
    app.add_source_parser(MarimoParser)
    app.add_directive("marimo-output", MarimoOutput)
    app.add_directive("marimo-launch", MarimoLaunch)
    app.connect("config-inited", exclude_other_python)
    app.connect("builder-inited", confine_documents)
    return {"version": "1", "parallel_read_safe": False, "parallel_write_safe": True}
