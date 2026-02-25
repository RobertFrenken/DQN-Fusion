<script>
  import { onMount, onDestroy } from 'svelte';
  import { ScatterChart } from '../lib/d3/ScatterChart.js';

  let { data = [], options = {}, height = 400 } = $props();

  let containerEl;
  let chart;

  onMount(() => {
    chart = new ScatterChart(containerEl, { height });
    if (data.length > 0) {
      chart.update(data, options);
    }
  });

  onDestroy(() => {
    if (chart) {
      chart.destroy();
    }
  });

  $effect(() => {
    if (chart && data.length > 0) {
      chart.update(data, options);
    }
  });
</script>

<div bind:this={containerEl} class="d3-scatter-container"></div>

<style>
  .d3-scatter-container {
    width: 100%;
    min-height: 200px;
  }

  /* D3-generated elements need :global() in Svelte */
  .d3-scatter-container :global(svg) {
    display: block;
  }

  .d3-scatter-container :global(.axis text) {
    fill: #374151;
    font-size: 11px;
  }

  .d3-scatter-container :global(.axis line),
  .d3-scatter-container :global(.axis path) {
    stroke: #9ca3af;
  }

  .d3-scatter-container :global(.grid line) {
    stroke: #e5e7eb;
    stroke-opacity: 0.7;
  }

  .d3-scatter-container :global(.grid path) {
    stroke: none;
  }

  .d3-scatter-container :global(.no-data) {
    color: #6b7280;
    text-align: center;
    padding: 2rem;
    font-style: italic;
  }
</style>

<!-- Tooltip styled globally (appended to body by D3) -->
<svelte:head>
  <style>
    .tooltip {
      position: absolute;
      background: #1f2937;
      color: #f9fafb;
      padding: 8px 12px;
      border-radius: 6px;
      font-size: 13px;
      line-height: 1.4;
      pointer-events: none;
      opacity: 0;
      transition: opacity 0.15s;
      z-index: 1000;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }
  </style>
</svelte:head>
