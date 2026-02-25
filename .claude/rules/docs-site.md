---
paths:
  - "docs-site/**"
---

# KD-GAT Docs Site (Astro + Svelte)

Astro 5 static site with Svelte 5 islands for interactive figures.

## Architecture

- Dual renderer: `figures.ts` registry (with `renderer: 'plot' | 'd3'` per figure) → `FigureIsland.astro` → either `PlotFigure.svelte` (Observable Plot) or `D3Chart.svelte` (D3)
- All 11 D3 chart types adapted with `import * as d3 from 'd3'`
- Hybrid data pipeline: Content Collections with Zod schemas (build-time), client-side fetch (runtime)
- CSS Grid layout (Distill-inspired: `.l-body`/`.l-wide`/`.l-full`/`.l-margin`)
- KaTeX for math (server-side via remark-math + rehype-katex)
- Interactive figures use Svelte 5 runes for state + `resource.svelte.ts` for reactive fetch

## Commands

```bash
module load node-js/22.12.0
cd docs-site && npm run dev      # Dev server at localhost:4321
cd docs-site && npm run build    # Static build → docs-site/dist/
scripts/sync-data.sh             # Sync dashboard data → src/data/
```

## Conventions

- **Runtime**: Node.js 22.12.0, npm only (no pnpm/yarn)
- **D3 imports**: `import * as d3 from 'd3'` (npm ES module, never CDN)
- **Svelte + D3**: Use `client:only="svelte"` for DOM-touching components (never `client:load`)
- **File organization**: D3 chart classes in `src/lib/d3/`, Svelte wrappers in `src/components/`, pages in `src/pages/`
- **Login node safe**: `npm install` and `npm run build` are fine on login nodes
- Deno Jupyter notebooks (`notebooks/deno_plot_template.ipynb`) for prototyping Observable Plot figures
