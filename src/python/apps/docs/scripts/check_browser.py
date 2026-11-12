"""Inspect a built STATADIC site in installed Chrome and retain visual evidence."""

import argparse
import asyncio
import functools
import hashlib
import http.server
import json
import shutil
import threading
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[5]
SITE = ROOT / "src/python/apps/docs/build/html"
OUT = ROOT / "build/ci/browser"
LOGO = ROOT / "src/python/apps/docs/static/brand/statadic-curve-fit-animated.svg"
LOGO_SHA256 = "e89b2d81fdeafad25d5fed777dd0a265a2545ada4a38e895be68ed708cdac332"
NAVIGATION = (
    ("Start", "start/index.html"),
    ("Tutorials", "tutorials/index.html"),
    ("How-to", "guides/index.html"),
    ("Concepts", "concepts/index.html"),
    ("Reference", "reference/index.html"),
    ("Evidence", "validation/index.html"),
    ("Blog", "blog/index.html"),
)
PAGES = (
    *(path for _, path in NAVIGATION),
    "start/first-regression.html",
    "tutorials/ols/index.html",
    "tutorials/ols/fit.html",
    "tutorials/ols/diagnose.html",
    "tutorials/ols/infer.html",
    "tutorials/ols/report.html",
    "reference/reporting.html",
    "reference/ordination.html",
    "reference/estimand.html",
    "reference/composition.html",
    "reference/conditional-laws.html",
    "reference/regression.html",
    "reference/hypotheses.html",
    "validation/references.html",
    "reference/regression.html",
    "formal/claims/STATADIC-LINEAR-004.html",
    "reference/api.html",
    "concepts/geometry.html",
    "tutorials/permutation-tests.html",
    "validation/research.html",
    "search.html",
)


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.targets = set()

    def handle_starttag(self, tag, attrs):
        self.targets.update(v for k, v in attrs if v and k == "id")
        if tag == "a":
            self.links.extend(v for k, v in attrs if k == "href" and v)
            self.targets.update(v for k, v in attrs if k == "name" and v)


def local_link_failures(site: Path) -> list[tuple[str, str]]:
    """Check local files and decoded HTML fragments, including same-page links."""
    site = site.resolve()
    documents = {}
    for path in site.rglob("*.html"):
        parsed = Links()
        parsed.feed(path.read_text())
        documents[path] = parsed
    missing = []
    for path, parsed in documents.items():
        for href in parsed.links:
            url = urlsplit(href)
            if url.scheme or url.netloc:
                continue
            target = path
            if url.path:
                target = (
                    site / unquote(url.path).lstrip("/")
                    if url.path.startswith("/")
                    else path.parent / unquote(url.path)
                ).resolve()
            if target.is_dir():
                target /= "index.html"
            if not target.is_relative_to(site) or not target.is_file():
                missing.append((str(path.relative_to(site)), href))
                continue
            fragment = unquote(url.fragment).split(":~:text=", 1)[0]
            if (
                fragment
                and target.suffix == ".html"
                and fragment not in documents[target].targets
            ):
                missing.append((str(path.relative_to(site)), href))
    return missing


class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


async def inspect_page(page, width, target):
    """Check readable typography and the rendered article, not CSS source text."""
    await page.evaluate(
        """async () => {
          await window.MathJax?.startup?.promise;
          await Promise.all([
            document.fonts.load('18px "STATADIC Roman"'),
            document.fonts.load('700 32px "STATADIC Roman"'),
            document.fonts.load('italic 18px "STATADIC Roman"'),
            document.fonts.load('14px "STATADIC Mono"')
          ]);
          await document.fonts.ready;
        }"""
    )
    appearance = await page.evaluate(
        r"""() => {
          const root = document.documentElement;
          const article = document.querySelector('.yue');
          const heading = article.querySelector('h1');
          const prose = article.querySelector('p') || article;
          // Contents entries are navigation; prose links retain underlines.
          const link = [...article.querySelectorAll('a:not(.headerlink)')]
            .find(e => !e.closest('.toctree-wrapper'));
          const articleStyle = getComputedStyle(article);
          const backgrounds = [...document.querySelectorAll(
            'body, .sy-head, .sy-foot, .sy-content, .sy-lside, .sy-rside'
          )].map(e => getComputedStyle(e).backgroundColor);
          const decorations = [...document.querySelectorAll('.sy-page, .sy-content')]
            .flatMap(e => ['::before', '::after'].map(pseudo =>
              getComputedStyle(e, pseudo).backgroundImage));
          return {
            mode: root.dataset.colorMode,
            light: root.classList.contains('light') && !root.classList.contains('dark'),
            overflow: root.scrollWidth > root.clientWidth,
            backgrounds,
            fonts: [...document.querySelectorAll(
              '.yue, .yue h1, .sy-head-brand, .globaltoc'
            )].map(e => getComputedStyle(e).fontFamily),
            romanLoaded: document.fonts.check('18px "STATADIC Roman"') &&
              document.fonts.check('700 32px "STATADIC Roman"') &&
              document.fonts.check('italic 18px "STATADIC Roman"'),
            monoLoaded: document.fonts.check('14px "STATADIC Mono"'),
            proseColor: getComputedStyle(prose).color,
            headingColors: [...article.querySelectorAll('h1,h2,h3,h4,h5,h6')]
              .map(e => getComputedStyle(e).color),
            linkColor: link ? getComputedStyle(link).color : null,
            linkDecoration: link ? getComputedStyle(link).textDecorationLine : null,
            fontSize: parseFloat(articleStyle.fontSize),
            lineHeight: parseFloat(articleStyle.lineHeight),
            articleWidth: article.getBoundingClientRect().width,
            headingSize: heading ?
              parseFloat(getComputedStyle(heading).fontSize) : null,
            rootFontSize: parseFloat(getComputedStyle(root).fontSize),
            codeFonts: [...article.querySelectorAll('pre')]
              .map(e => getComputedStyle(e).fontFamily),
            codeColors: [...article.querySelectorAll('.highlight pre span')]
              .map(e => getComputedStyle(e).color),
            decorations,
            decorativeImages: [...document.querySelectorAll('img')]
              .filter(e => /(?:margin|hedera|workflow)\.svg(?:$|[#?])/.test(e.src))
              .map(e => e.src)
          };
        }"""
    )
    assert appearance["mode"] == "light" and appearance["light"], (target, appearance)
    assert not appearance["overflow"], (width, target)
    assert set(appearance["backgrounds"]) == {"rgb(255, 255, 255)"}, target
    assert all("STATADIC Roman" in font for font in appearance["fonts"]), target
    assert appearance["romanLoaded"], target
    assert appearance["monoLoaded"], target
    assert appearance["proseColor"] == "rgb(0, 0, 0)", (target, appearance)
    assert set(appearance["headingColors"]) <= {"rgb(0, 0, 0)"}, target
    if appearance["linkColor"] is not None:
        assert appearance["linkColor"] == "rgb(29, 63, 143)", target
        assert "underline" in appearance["linkDecoration"], target
    assert appearance["fontSize"] >= 16, target
    assert appearance["lineHeight"] >= 1.5 * appearance["fontSize"], target
    assert 0 < appearance["articleWidth"] <= width, (target, appearance)
    if appearance["headingSize"] is not None:
        assert appearance["headingSize"] <= 2.5 * appearance["rootFontSize"], target
    assert all("STATADIC Mono" in font for font in appearance["codeFonts"]), target
    assert set(appearance["codeColors"]) <= {"rgb(0, 0, 0)"}, target
    assert set(appearance["decorations"]) == {"none"}, target
    assert not appearance["decorativeImages"], target
    assert await page.locator(".theme-switch").count() == 0
    logo = page.locator(".sy-head-brand img.light-logo")
    assert await logo.count() == 1 and await logo.is_visible(), target
    assert not await page.locator(".sy-head-brand img.dark-logo").is_visible(), target
    assert await page.locator(".sy-head-brand strong").is_visible(), target
    assert await page.locator(".sy-head-brand strong").inner_text() == "STATADIC", (
        target
    )
    brand = await logo.evaluate(
        """async image => {
          await image.decode();
          const box = image.getBoundingClientRect();
          return {
            complete: image.complete,
            naturalWidth: image.naturalWidth,
            naturalHeight: image.naturalHeight,
            width: box.width,
            height: box.height,
            alt: image.alt,
            source: image.src
          };
        }"""
    )
    assert brand["complete"] and brand["naturalWidth"] == brand["naturalHeight"] == 192
    assert brand["width"] == brand["height"] == 28, (target, brand)
    assert brand["alt"] == "STATADIC", target
    assert urlsplit(brand["source"]).path.endswith("/" + LOGO.name), target
    favicon = page.locator('link[rel~="icon"]')
    assert await favicon.count() == 1, target
    favicon_url = await favicon.evaluate("link => link.href")
    assert urlsplit(favicon_url).path.endswith("/" + LOGO.name), target
    if target == "index.html":
        for source in {brand["source"], favicon_url}:
            response = await page.request.get(source)
            assert response.ok, source
            assert hashlib.sha256(await response.body()).hexdigest() == LOGO_SHA256, (
                source
            )
        brand["sha256"] = LOGO_SHA256
    appearance["brand"] = brand | {"favicon": favicon_url}
    chapter_pages = {"tutorials/ols/diagnose.html", "tutorials/ols/report.html"}
    if target in chapter_pages:
        figure = page.locator("img.marimo-figure").first
        assert await figure.count() == 1, target
        loaded = await figure.evaluate(
            """async image => {
              await image.decode();
              return {
                complete: image.complete,
                width: image.naturalWidth,
                height: image.naturalHeight,
                alt: image.alt
              };
            }"""
        )
        assert loaded["complete"] and loaded["width"] > 0 and loaded["height"] > 0, (
            target,
            loaded,
        )
        assert loaded["alt"].strip(), target
        appearance["chapterFigure"] = loaded
    return appearance


async def inspect_navigation(page, url, width):
    """Exercise navigation, nested sections and search using the keyboard."""
    await page.goto(url + "/index.html", wait_until="load")
    if width < 768:
        menu = page.locator('button[aria-controls="head-nav"]')
        await menu.focus()
        await menu.press("Enter")
        assert await menu.get_attribute("aria-expanded") == "true"
    first_link = page.locator(".sy-head-links a").first
    await page.keyboard.press("Tab")
    await first_link.focus()
    assert await first_link.evaluate("e => e.matches(':focus-visible')")
    assert await first_link.evaluate("e => getComputedStyle(e).outlineWidth") == "2px"
    await first_link.press("Enter")
    await page.wait_for_url("**/start/index.html")
    await page.goto(url + "/tutorials/ols/index.html", wait_until="load")
    for sidebar, breakpoint in [("lside", 768), ("rside", 1280)]:
        if width < breakpoint:
            control = page.locator(f'.sy-breadcrumbs button[aria-controls="{sidebar}"]')
            await control.focus()
            await control.press("Enter")
            assert await page.locator(f"#{sidebar}").get_attribute("aria-hidden") == (
                "false"
            )
            await control.press("Enter")
            assert await page.locator(f"#{sidebar}").get_attribute("aria-hidden") == (
                "true"
            )
    if width < 768:
        await page.locator('.sy-breadcrumbs button[aria-controls="lside"]').press(
            "Enter"
        )
    nested = page.locator(".globaltoc li > button:visible").first
    assert await nested.count() == 1
    await nested.focus()
    before = await nested.get_attribute("aria-label")
    await nested.press("Enter")
    assert await nested.get_attribute("aria-label") != before
    await nested.press("Enter")
    await page.goto(url + "/index.html", wait_until="load")
    if width < 768:
        await page.locator('button[aria-controls="head-nav"]').press("Enter")
    search = page.locator(".searchbox input")
    await search.fill("regression")
    await search.press("Enter")
    await page.wait_for_url("**/search.html?q=regression")
    await page.locator("#search-results li a").first.wait_for()


async def inspect_text_resize(browser, url, errors):
    """Retain a narrow-screen check with all text enlarged to 200 percent."""
    context = await browser.new_context(viewport={"width": 390, "height": 844})
    page = await context.new_page()
    page.on("pageerror", lambda error: errors.append(str(error)))
    observations = []
    try:
        for target in ("tutorials/ols/report.html", "reference/reporting.html"):
            await page.goto(url + "/" + target, wait_until="load")
            original = await page.evaluate(
                "parseFloat(getComputedStyle(document.documentElement).fontSize)"
            )
            await page.evaluate(
                "size => document.documentElement.style.fontSize = `${size * 2}px`",
                original,
            )
            appearance = await inspect_page(page, 390, target)
            assert appearance["rootFontSize"] == original * 2
            observations.append({"target": target, "appearance": appearance})
            name = target.removesuffix(".html").replace("/", "-")
            await page.screenshot(
                path=str(OUT / f"{name}-390-text-200.png"), full_page=True
            )
    finally:
        await context.close()
    return observations


async def inspect_reduced_motion(browser, url, errors):
    """Hide the animated header image while retaining its wordmark and link."""
    context = await browser.new_context(reduced_motion="reduce")
    try:
        page = await context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        await page.goto(url + "/index.html", wait_until="load")
        assert await page.evaluate(
            "matchMedia('(prefers-reduced-motion: reduce)').matches"
        )
        images = page.locator(".sy-head-brand img")
        assert await images.count() == 2
        assert not any([await image.is_visible() for image in await images.all()])
        brand = page.locator(".sy-head-brand")
        wordmark = brand.locator("strong")
        assert await wordmark.is_visible()
        assert await wordmark.inner_text() == "STATADIC"
        assert await brand.get_attribute("href") == "#"
        return {"preference": "reduce", "images_visible": False, "wordmark": "STATADIC"}
    finally:
        await context.close()


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--browser", default=shutil.which("google-chrome") or shutil.which("chromium")
    )
    args = parser.parse_args()
    if not args.browser:
        parser.error("Install Chrome/Chromium or pass --browser /path/to/executable")
    # A receipt describes this run, never screenshots left by an older site.
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    assert hashlib.sha256(LOGO.read_bytes()).hexdigest() == LOGO_SHA256
    assert not list(SITE.rglob("margin.svg"))
    assert not list(SITE.rglob("hedera.svg"))
    missing = local_link_failures(SITE)
    assert not missing, missing
    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(Handler, directory=str(SITE))
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}"
    result = {
        "html_targets": len(list(SITE.rglob("*.html"))),
        "layouts": [],
        "javascript_errors": [],
    }
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                executable_path=args.browser, headless=True, args=["--no-sandbox"]
            )
            result["browser_version"] = browser.version
            for mode in ["light", "dark"]:
                for width, height in [
                    (1440, 1000),
                    (1280, 1000),
                    (1024, 900),
                    (768, 900),
                    (390, 844),
                    (320, 720),
                ]:
                    context = await browser.new_context(
                        viewport={"width": width, "height": height}, color_scheme=mode
                    )
                    page = await context.new_page()
                    page.on(
                        "pageerror",
                        lambda e: result["javascript_errors"].append(str(e)),
                    )
                    requests = []
                    page.on(
                        "request", lambda r, requests=requests: requests.append(r.url)
                    )
                    await page.goto(url + "/index.html", wait_until="load")
                    await page.evaluate("document.fonts.ready")
                    assert await page.locator(
                        ".sy-head-links a"
                    ).all_text_contents() == [title for title, _ in NAVIGATION]
                    assert await page.locator(".sy-head-links a").evaluate_all(
                        "links => links.map(link => link.getAttribute('href'))"
                    ) == [path for _, path in NAVIGATION]
                    appearance = await inspect_page(page, width, "index.html")
                    assert not any(
                        "fonts.googleapis.com" in u or "fonts.gstatic.com" in u
                        for u in requests
                    )
                    await page.screenshot(
                        path=str(OUT / f"home-{width}-{mode}.png"), full_page=True
                    )
                    result["layouts"].append(
                        {
                            "width": width,
                            "mode": mode,
                            "appearance": appearance,
                        }
                    )
                    for target in PAGES:
                        response = await page.goto(
                            url + "/" + target, wait_until="load"
                        )
                        assert response.status == 200, target
                        await inspect_page(page, width, target)
                        if target == "start/first-regression.html":
                            launch = page.locator(".marimo-launch")
                            assert await launch.count() == 1
                            badge = launch.locator('a[href*="molab.marimo.io/github/"]')
                            assert await badge.count() == 1
                            assert await page.locator("#introduction").count() == 1
                            assert await page.locator("#appendix").count() == 0
                        if width in (1440, 390):
                            name = target.removesuffix(".html").replace("/", "-")
                            await page.screenshot(
                                path=str(OUT / f"{name}-{width}-{mode}.png")
                            )
                    await page.goto(
                        url + "/formal/claims/STATADIC-LINEAR-004.html",
                        wait_until="load",
                    )
                    await page.locator("mjx-container").first.wait_for(timeout=30000)
                    assert (
                        "pending" in (await page.locator("body").inner_text()).lower()
                    )
                    if mode == "light":
                        await inspect_navigation(page, url, width)
                        await page.goto(
                            url + "/validation/research.html",
                            wait_until="load",
                        )
                        ownership = page.locator("main details").first
                        assert await ownership.count() == 1
                        assert not await ownership.evaluate("e => e.open")
                        await ownership.locator("summary").focus()
                        await page.keyboard.press("Enter")
                        assert await ownership.evaluate("e => e.open")
                    await context.close()
            result["text_resize"] = await inspect_text_resize(
                browser, url, result["javascript_errors"]
            )
            result["reduced_motion"] = await inspect_reduced_motion(
                browser, url, result["javascript_errors"]
            )
            # Previously stored preferences cannot restore the removed dark theme.
            for saved in ["dark", "auto"]:
                context = await browser.new_context(color_scheme="dark")
                await context.add_init_script(f"localStorage._theme = {saved!r}")
                page = await context.new_page()
                for target in ["index.html", "start/first-regression.html"]:
                    await page.goto(url + "/" + target, wait_until="load")
                    await inspect_page(page, 1280, target)
                await context.close()
            await browser.close()
        assert not result["javascript_errors"], result["javascript_errors"]
        result["screenshots"] = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in OUT.glob("*.png")
        }
        (OUT / "report.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    asyncio.run(main())
