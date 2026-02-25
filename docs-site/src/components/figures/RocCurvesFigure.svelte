<script>
  /**
   * Interactive ROC curves figure.
   * Controls: dataset, scale.
   * Fetches per-run ROC curve JSON for VGAE, GAT, and DQN-Fusion,
   * then combines them into a multi-series view.
   */
  import D3Chart from '../D3Chart.svelte';

  let { datasets = ['hcrl_ch', 'hcrl_sa', 'set_01', 'set_02', 'set_03', 'set_04'] } = $props();

  let dataset = $state('hcrl_sa');
  let scale = $state('large');

  let data = $state(null);
  let loading = $state(false);
  let error = $state(null);

  const MODELS = ['vgae', 'gat', 'fusion'];

  $effect(() => {
    const runId = `${dataset}_eval_${scale}_evaluation`;
    loading = true;
    error = null;
    const ctrl = new AbortController();

    // Fetch all three model ROC curves in parallel
    Promise.all(
      MODELS.map(model =>
        fetch(`/data/roc_curves/${runId}_${model}.json`, { signal: ctrl.signal })
          .then(r => r.ok ? r.json() : null)
          .then(raw => raw ? { name: model.toUpperCase(), ...((raw.data ?? raw)) } : null)
          .catch(() => null)
      )
    ).then(results => {
      const series = results.filter(Boolean);
      if (series.length === 0) {
        error = 'No ROC data found for this configuration';
      } else {
        data = { series, _xLabel: 'False Positive Rate', _yLabel: 'True Positive Rate', _refLine: 'diagonal' };
      }
      loading = false;
    }).catch(e => {
      if (e.name !== 'AbortError') { error = e.message; loading = false; }
    });

    return () => ctrl.abort();
  });

  let chartOptions = $derived({
    xLabel: 'False Positive Rate',
    yLabel: 'True Positive Rate',
    refLine: 'diagonal',
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
    Scale
    <select bind:value={scale}>
      <option value="large">Large</option>
      <option value="small">Small</option>
    </select>
  </label>
</div>

{#if loading}
  <div class="figure-status">Loading ROC curves...</div>
{:else if error}
  <div class="figure-status figure-error">{error}</div>
{:else if data}
  <D3Chart chartType="curve" {data} options={chartOptions} height={400} />
{:else}
  <div class="figure-status">Select options to load ROC curves.</div>
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
