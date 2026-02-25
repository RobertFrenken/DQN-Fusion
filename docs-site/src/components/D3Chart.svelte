<script>
  import { onMount, onDestroy } from 'svelte';

  let { chartType, data = null, options = {}, height = 400, dataSource = null } = $props();

  // If dataSource is provided (URL path), fetch data on mount
  let fetchedData = $state(null);

  let containerEl;
  let chart;

  // Use import.meta.glob to discover chart classes at build time.
  // Only files that exist in src/lib/d3/ are included — no build errors for missing charts.
  const modules = import.meta.glob('../lib/d3/*Chart.js');
  const graphModules = import.meta.glob('../lib/d3/ForceGraph.js');

  // Map chartType names to their module paths
  const TYPE_TO_PATH = {
    scatter:   '../lib/d3/ScatterChart.js',
    bar:       '../lib/d3/BarChart.js',
    line:      '../lib/d3/LineChart.js',
    histogram: '../lib/d3/HistogramChart.js',
    heatmap:   '../lib/d3/HeatmapChart.js',
    curve:     '../lib/d3/CurveChart.js',
    force:     '../lib/d3/ForceGraph.js',
    timeline:  '../lib/d3/TimelineChart.js',
    bubble:    '../lib/d3/BubbleChart.js',
    table:     '../lib/d3/TableChart.js',
    summary:   '../lib/d3/SummaryChart.js',
  };

  function getLoader(type) {
    const path = TYPE_TO_PATH[type];
    if (!path) return null;
    return modules[path] || graphModules[path] || null;
  }

  function hasData(d) {
    if (d == null) return false;
    if (Array.isArray(d)) return d.length > 0;
    if (typeof d === 'object') return Object.keys(d).length > 0;
    return false;
  }

  onMount(async () => {
    const loader = getLoader(chartType);
    if (!loader) {
      console.warn(`D3Chart: chart type "${chartType}" not available (module not found)`);
      return;
    }

    try {
      const module = await loader();
      // Resolve the default export or the first named export (skip __esModule)
      const ChartClass = module.default
        || module[Object.keys(module).find(k => k !== '__esModule')];

      if (!ChartClass) {
        console.error(`D3Chart: no chart class found in module for "${chartType}"`);
        return;
      }

      chart = new ChartClass(containerEl, { height });

      // If dataSource is provided, fetch it
      if (dataSource && !hasData(data)) {
        try {
          const res = await fetch(`/data/${dataSource}`);
          const raw = await res.json();
          fetchedData = raw.data ?? raw;
        } catch (e) {
          console.error(`D3Chart: failed to fetch "${dataSource}":`, e.message);
        }
      }

      const d = hasData(data) ? data : fetchedData;
      if (hasData(d)) chart.update(d, options);
    } catch (err) {
      console.error(`D3Chart: failed to load "${chartType}":`, err.message);
    }
  });

  onDestroy(() => {
    chart?.destroy();
  });

  $effect(() => {
    const d = hasData(data) ? data : fetchedData;
    if (chart && hasData(d)) {
      chart.update(d, options);
    }
  });
</script>

<div bind:this={containerEl} class="d3-chart-container" style:min-height="{height}px"></div>

<style>
  .d3-chart-container {
    width: 100%;
    min-height: 200px;
  }
</style>
