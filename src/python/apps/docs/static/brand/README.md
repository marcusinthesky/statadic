# Logo and local fonts

`statadic-curve-fit-animated.svg` is the supplied STATADIC logo and favicon, copied
from `.tmp/statadic-curve-fit-animated.svg` with only the repository-required final
newline added. Its repository SHA-256 is
`6655ae244a8c060a7ff6d6f2686b66174e6ac98852b9b8999547c09d5a8191dd`.
The asset retains its original palette, animation, title, description and
reduced-motion stylesheet; site styling does not rewrite the SVG.
The supplied SMIL animation continues in Chrome despite that internal rule.
The site's reduced-motion wrapper therefore hides only the header image and
keeps the STATADIC wordmark and home link visible. Favicon animation depends on the
browser; this wrapper does not modify the asset or its standalone behavior.

The WOFF2 font subsets come from
[Florilegium](https://github.com/kevinraymond/florilegium), revision
`5c9bd05ad7ad0ed7b0bfb0cfa2adc1720406e63b`.

Latin Modern Roman supplies body, heading and navigation text; Latin Modern Mono
supplies code. The website serves these files locally, with black text on white
and Delft-blue (`#1d3f8f`) links and focus outlines.

The subsets use `fonts/GUST-FONT-LICENSE.txt`. Preserve that notice when
redistributing the fonts. The local CSS family aliases do not modify the files.
