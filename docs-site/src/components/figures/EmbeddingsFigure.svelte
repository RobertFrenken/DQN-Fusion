<script>
  /**
   * Interactive VGAE/GAT latent space embeddings figure.
   * Controls: dataset, model, projection method, overlay mode.
   * Fetches per-run embedding JSON on demand.
   */
  import D3Chart from '../D3Chart.svelte';

  let { datasets = ['hcrl_ch', 'hcrl_sa', 'set_01', 'set_02', 'set_03', 'set_04'] } = $props();

  // Reload triggers (changing these fetches new data)
  let dataset = $state('hcrl_sa');
  let model = $state('vgae');
  let method = $state('umap');
  let scale = $state('large');

  // Display option (no refetch, just re-renders)
  let overlay = $state('points');

  // Fetched data
  let data = $state(null);
  let loading = $state(false);
  let error = $state(null);

  // Fetch embedding data when reload triggers change
  $effect(() => {
    const runId = `${dataset}_eval_${scale}_evaluation`;
    const url = `/data/embeddings/${runId}_${model}_${method}.json`;
    loading = true;
    error = null;
    const ctrl = new AbortController();

    fetch(url, { signal: ctrl.signal })
      .then(r => {
        if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
        return r.json();
      })
      .then(raw => { data = raw.data ?? raw; loading = false; })
      .catch(e => {
        if (e.name !== 'AbortError') { error = e.message; loading = false; }
      });

    return () => ctrl.abort();
  });

  // Chart options (derived, no refetch)
  let chartOptions = $derived({
    xField: 'dim0',
    yField: 'dim1',
    colorField: 'label',
    xLabel: `${method.toUpperCase()} Dimension 1`,
    yLabel: `${method.toUpperCase()} Dimension 2`,
    labelColors: true,
    showDensity: overlay === 'density' ? true : overlay === 'voronoi' ? 'voronoi' : false,
  });
</script>

<div class="figure-controls">
  <label>
    Dataset
    <select bind:value={dataset}>
      {#each datasets as ds}
        <option value={ds}>{ds}</option>
      {/each}
    </select>
  </label>
  <label>
    Model
    <select bind:value={model}>
      <option value="vgae">VGAE</option>
      <option value="gat">GAT</option>
    </select>
  </label>
  <label>
    Projection
    <select bind:value={method}>
      <option value="umap">UMAP</option>
      <option value="pymde">PyMDE</option>
    </select>
  </label>
  <label>
    Scale
    <select bind:value={scale}>
      <option value="large">Large</option>
      <option value="small">Small</option>
    </select>
  </label>
  <label>
    Overlay
    <select bind:value={overlay}>
      <option value="points">Points Only</option>
      <option value="density">+ Density</option>
      <option value="voronoi">+ Voronoi</option>
    </select>
  </label>
</div>

{#if loading}
  <div class="figure-status">Loading embeddings...</div>
{:else if error}
  <div class="figure-status figure-error">Error: {error}</div>
{:else if data}
  <D3Chart chartType="scatter" {data} options={chartOptions} height={420} />
{:else}
  <div class="figure-status">Select options to load embeddings.</div>
{/if}

<style>
  .figure-controls {
    display: flex;
    flex-wrap: wrap;
    gap: 0.75rem;
    margin-bottom: 1rem;
    padding: 0.75rem 1rem;
    background: #f8f9fa;
    border-radius: 8px;
    border: 1px solid #e2e8f0;
  }

  .figure-controls label {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
    font-size: 0.8rem;
    font-weight: 600;
    color: #64748b;
  }

  .figure-controls select {
    padding: 0.35rem 0.5rem;
    border: 1px solid #d1d5db;
    border-radius: 4px;
    font-size: 0.85rem;
    background: white;
    color: #1a1a2e;
  }

  .figure-status {
    text-align: center;
    padding: 2rem;
    color: #64748b;
    font-style: italic;
  }

  .figure-error {
    color: #dc2626;
  }
</style>
