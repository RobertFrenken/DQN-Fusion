<script>
  /**
   * Interactive training curves figure.
   * Controls: dataset, model, scale, metric(s).
   * Fetches per-run training curve JSON on demand.
   */
  import D3Chart from '../D3Chart.svelte';

  let { datasets = ['hcrl_ch', 'hcrl_sa', 'set_01', 'set_02', 'set_03', 'set_04'] } = $props();

  // Reload triggers
  let dataset = $state('hcrl_sa');
  let model = $state('gat');
  let scale = $state('large');
  let stage = $state('curriculum');

  // Display option
  let metric = $state('val_loss');

  // Fetched data
  let data = $state(null);
  let loading = $state(false);
  let error = $state(null);

  $effect(() => {
    const runId = `${dataset}_${model}_${scale}_${stage}`;
    const url = `/data/training_curves/${runId}.json`;
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

  let chartOptions = $derived({
    metric,
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
    Scale
    <select bind:value={scale}>
      <option value="large">Large</option>
      <option value="small">Small</option>
    </select>
  </label>
  <label>
    Stage
    <select bind:value={stage}>
      <option value="autoencoder">Autoencoder</option>
      <option value="curriculum">Curriculum</option>
    </select>
  </label>
  <label>
    Metric
    <select bind:value={metric}>
      <option value="val_loss">Validation Loss</option>
      <option value="train_loss">Training Loss</option>
      <option value="val_loss,train_loss">Both</option>
    </select>
  </label>
</div>

{#if loading}
  <div class="figure-status">Loading training curves...</div>
{:else if error}
  <div class="figure-status figure-error">Error: {error}</div>
{:else if data}
  <D3Chart chartType="line" {data} options={chartOptions} height={380} />
{:else}
  <div class="figure-status">Select options to load curves.</div>
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
