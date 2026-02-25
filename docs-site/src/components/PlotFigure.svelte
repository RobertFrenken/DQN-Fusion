<script>
  /**
   * PlotFigure — Generic Observable Plot wrapper for Svelte 5.
   *
   * Instead of 150-line D3 chart classes, describe your chart declaratively:
   *
   *   <PlotFigure
   *     data={points}
   *     marks={[{ type: 'dot', x: 'fpr', y: 'tpr', fill: 'model' }]}
   *     options={{ grid: true, color: { legend: true } }}
   *   />
   *
   * Mark types map to Observable Plot marks:
   *   dot, line, barY, barX, rectY, rectX, cell, text, ruleX, ruleY, areaY, tickX, tickY
   */
  import * as Plot from '@observablehq/plot';

  let {
    data = [],
    marks: markConfigs = [],
    options = {},
    height = 400,
    width = undefined,
  } = $props();

  let container = $state();

  // Map string mark types to Plot functions
  const MARK_MAP = {
    dot: Plot.dot,
    line: Plot.line,
    lineY: Plot.lineY,
    lineX: Plot.lineX,
    barY: Plot.barY,
    barX: Plot.barX,
    rectY: Plot.rectY,
    rectX: Plot.rectX,
    cell: Plot.cell,
    text: Plot.text,
    ruleX: Plot.ruleX,
    ruleY: Plot.ruleY,
    ruleY_value: (data, opts) => Plot.ruleY(data, opts),
    areaY: Plot.areaY,
    tickX: Plot.tickX,
    tickY: Plot.tickY,
    link: Plot.link,
    frame: () => Plot.frame(),
  };

  function buildMark(cfg) {
    const { type, data: markData, ...markOpts } = cfg;
    const fn = MARK_MAP[type];
    if (!fn) {
      console.warn(`PlotFigure: unknown mark type "${type}"`);
      return null;
    }

    // Some marks like ruleY([0.5]) take an array of values, not the main data
    if (markData !== undefined) {
      return fn(markData, markOpts);
    }
    // frame() takes no args
    if (type === 'frame') {
      return fn();
    }
    return fn(data, markOpts);
  }

  $effect(() => {
    if (!container) return;

    // Clear previous plot
    while (container.firstChild) container.firstChild.remove();

    const builtMarks = markConfigs.map(buildMark).filter(Boolean);
    if (builtMarks.length === 0 && data.length === 0) return;

    try {
      const plot = Plot.plot({
        height,
        width: width ?? undefined,
        ...options,
        marks: builtMarks,
      });
      container.append(plot);
    } catch (err) {
      console.error('PlotFigure render error:', err.message);
    }
  });
</script>

<div bind:this={container} class="plot-figure-container" style:min-height="{height}px"></div>

<style>
  .plot-figure-container {
    width: 100%;
    min-height: 200px;
  }
  .plot-figure-container :global(svg) {
    max-width: 100%;
    height: auto;
  }
  /* Observable Plot default styles integration */
  .plot-figure-container :global(figure) {
    margin: 0;
  }
</style>
