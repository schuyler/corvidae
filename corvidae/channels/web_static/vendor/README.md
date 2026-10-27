# Vendored browser libraries

The web transport's UI renders model output as Markdown. Both libraries are
served from this directory, never from a CDN: the page's Content Security
Policy allows scripts from `'self'` only.

| File | Library | Version | Licence | Source |
|---|---|---|---|---|
| `marked.min.js` | [marked](https://github.com/markedjs/marked) | 15.0.12 | MIT | https://cdn.jsdelivr.net/npm/marked@15.0.12/marked.min.js |
| `purify.min.js` | [DOMPurify](https://github.com/cure53/DOMPurify) | 3.2.6 | Apache-2.0 or MPL-2.0 | https://cdn.jsdelivr.net/npm/dompurify@3.2.6/dist/purify.min.js |

sha256 of the downloaded files:

- `marked.min.js`: `3e7e7d7feb3e5d58cb6c804f68ab5c24cc7e5eb6270fd6e5cbb9124739217d0c`
  (unmodified)
- `purify.min.js`: `89e1fa7647cb495370d3a997ace4387f5d15d9f4c5af12352c53daa400956287`
  as downloaded. The trailing `//# sourceMappingURL=purify.min.js.map` line
  is removed (the map isn't shipped, and browsers would request it), which
  gives `fc42f9bf6f37e0df85e1a20ee4045983f89c1d04914f50c1113d8c4afe3a1bcb`.

Each file keeps its upstream licence banner. To upgrade, download the new
version, check its hash against the npm registry tarball, strip purify's
source-map line again, and update this table.
