/* KD-GAT Dashboard — Data loading and initialization */

(async function() {
    const BASE = 'data/';
    const EXPECTED_MAJOR_VERSION = '1';

    async function loadJSON(path) {
        try {
            const raw = await d3.json(BASE + path);
            if (!raw) return [];
            // Unwrap versioned envelope (backwards-compatible with old format)
            if (raw.schema_version != null) {
                const major = String(raw.schema_version).split('.')[0];
                if (major !== EXPECTED_MAJOR_VERSION) {
                    console.warn(`Schema version mismatch: expected major ${EXPECTED_MAJOR_VERSION}, got ${raw.schema_version}`);
                }
                return raw.data ?? raw;
            }
            return raw;
        } catch (e) {
            console.warn(`Failed to load ${path}:`, e.message);
            return [];
        }
    }

    // Load all data files in parallel
    const [leaderboard, runs, datasets, kdTransfer, metricCatalog] = await Promise.all([
        loadJSON('leaderboard.json'),
        loadJSON('runs.json'),
        loadJSON('datasets.json'),
        loadJSON('kd_transfer.json'),
        loadJSON('metrics/metric_catalog.json'),
    ]);

    // --- Populate metric dropdowns dynamically ---
    function populateMetricSelect(selectId, defaultMetrics, selectedDefault) {
        const select = document.getElementById(selectId);
        if (!select) return;
        // Use catalog if available, else fall back to defaults
        const metrics = metricCatalog.length > 0 ? metricCatalog : defaultMetrics;
        select.innerHTML = '';
        metrics.forEach(m => {
            const opt = document.createElement('option');
            opt.value = m;
            opt.textContent = m.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
            if (m === selectedDefault) opt.selected = true;
            select.appendChild(opt);
        });
    }

    // Populate leaderboard and dataset comparison metric selects from catalog
    populateMetricSelect('lb-metric', ['f1', 'accuracy', 'precision', 'recall', 'auc', 'mcc'], 'f1');
    populateMetricSelect('dc-metric', ['f1', 'accuracy', 'auc'], 'f1');

    // --- Leaderboard ---
    const lbMetricSelect = document.getElementById('lb-metric');
    function renderLeaderboard() {
        Charts.leaderboard('#lb-table-container', leaderboard, lbMetricSelect.value);
    }
    lbMetricSelect.addEventListener('change', renderLeaderboard);
    renderLeaderboard();

    // --- Dataset Comparison ---
    const dcMetricSelect = document.getElementById('dc-metric');
    function renderDatasetComparison() {
        Charts.datasetComparison('#dc-chart', leaderboard, dcMetricSelect.value);
    }
    dcMetricSelect.addEventListener('change', renderDatasetComparison);
    renderDatasetComparison();

    // --- KD Transfer ---
    Charts.kdTransfer('#kd-chart', kdTransfer);

    // --- Training Curves ---
    const tcMetricSelect = document.getElementById('tc-metric');
    const tcRunSelect = document.getElementById('tc-run');
    let curveCache = {};

    // Populate run selector from runs that have curve data
    // Try loading a listing or fall back to completed runs
    const curveRuns = runs.filter(r => r.status === 'complete' && r.stage !== 'evaluation');

    if (curveRuns.length === 0) {
        document.getElementById('tc-chart').innerHTML = '<div class="no-data">No training curve data available</div>';
    } else {
        curveRuns.forEach(r => {
            const opt = document.createElement('option');
            opt.value = r.run_id;
            opt.textContent = r.run_id;
            tcRunSelect.appendChild(opt);
        });

        async function renderTrainingCurves() {
            const runId = tcRunSelect.value;
            if (!runId) return;

            if (!curveCache[runId]) {
                const fname = runId.replace(/\//g, '_') + '.json';
                curveCache[runId] = await loadJSON('training_curves/' + fname);
            }
            Charts.trainingCurves('#tc-chart', curveCache[runId], tcMetricSelect.value);
        }

        tcMetricSelect.addEventListener('change', renderTrainingCurves);
        tcRunSelect.addEventListener('change', renderTrainingCurves);
        renderTrainingCurves();
    }

    // --- Run Timeline ---
    Charts.runTimeline('#rt-chart', runs);

    // --- Navigation ---
    document.querySelectorAll('nav a').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            document.querySelectorAll('nav a').forEach(l => l.classList.remove('active'));
            e.target.classList.add('active');
            // Show only the target panel
            const targetId = e.target.getAttribute('href').slice(1);
            document.querySelectorAll('main > section').forEach(s => {
                s.style.display = s.id === targetId ? '' : 'none';
            });
        });
    });
    // Show only the first panel on load
    document.querySelectorAll('main > section').forEach((s, i) => {
        s.style.display = i === 0 ? '' : 'none';
    });
})();
