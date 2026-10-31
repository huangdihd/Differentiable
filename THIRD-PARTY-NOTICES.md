# Third-party notices

Differentiable is licensed under the GNU General Public License, version 3 or later (see
`LICENSE`). It builds on the open-source work listed below. All of these licenses are compatible
with the GPLv3; none of them imposes terms beyond it on Differentiable itself.

The full license text of each package ships inside `node_modules/<package>/` after `npm install`
and is redistributed unchanged wherever that package is included in a build.

## Bundled at runtime

These packages are compiled into the browser bundle under `dist/` and are therefore conveyed to
every visitor of the demo build.

| Package | Version | License | Copyright |
| --- | --- | --- | --- |
| [react](https://github.com/facebook/react) | 19.x | MIT | Meta Platforms, Inc. and affiliates |
| [react-dom](https://github.com/facebook/react) | 19.x | MIT | Meta Platforms, Inc. and affiliates |
| [katex](https://github.com/KaTeX/KaTeX) | 0.19.x | MIT | Khan Academy and KaTeX contributors |
| [react-markdown](https://github.com/remarkjs/react-markdown) | 10.x | MIT | Titus Wormer and contributors |
| [remark-gfm](https://github.com/remarkjs/remark-gfm) | 4.x | MIT | Titus Wormer and contributors |
| [remark-math](https://github.com/remarkjs/remark-math) | 6.x | MIT | Junyoung Choi and contributors |
| [remark-breaks](https://github.com/remarkjs/remark-breaks) | 4.x | MIT | Titus Wormer and contributors |
| [rehype-katex](https://github.com/remarkjs/remark-math) | 7.x | MIT | Junyoung Choi and contributors |

`dist/` is produced by `npm run build:demo`, which also runs `javascript-obfuscator` over the
output. Obfuscation is a build step only; the corresponding source is this repository, as GPLv3
section 6 requires.

## Build and test tooling

Not included in the browser bundle, but required to build or verify the project.

| Package | Version | License |
| --- | --- | --- |
| [vite](https://github.com/vitejs/vite) | 6.x | MIT |
| [vitest](https://github.com/vitest-dev/vitest) | 5.x | MIT |
| [typescript](https://github.com/microsoft/TypeScript) | 5.8.x | Apache-2.0 |
| [prettier](https://github.com/prettier/prettier) | 3.x | MIT |
| [javascript-obfuscator](https://github.com/javascript-obfuscator/javascript-obfuscator) | 4.1.x | BSD-2-Clause |
| [@types/react](https://github.com/DefinitelyTyped/DefinitelyTyped) | 19.x | MIT |
| [@types/react-dom](https://github.com/DefinitelyTyped/DefinitelyTyped) | 19.x | MIT |

## Fonts

The app and the `demos/` pages load their typefaces from the Google Fonts API at runtime; no font
files are stored in this repository.

| Font | License |
| --- | --- |
| IBM Plex Mono | SIL Open Font License 1.1 |
| JetBrains Mono | SIL Open Font License 1.1 |
| Noto Sans SC | SIL Open Font License 1.1 |
| Noto Serif SC | SIL Open Font License 1.1 |
| Source Serif 4 | SIL Open Font License 1.1 |
| STIX Two Text | SIL Open Font License 1.1 |

## Content

All level data, the garden rhyme used as the language-model corpus, the generated music and the
graphics in `demos/` are original to this project. Dependencies are the only third-party material.
