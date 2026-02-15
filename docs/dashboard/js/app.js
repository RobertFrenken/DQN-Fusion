/* KD-GAT Dashboard — Data loading and initialization */

(async function() {
    const BASE = 'data/';

    async function loadJSON(path) {
        try {
            return await d3.json(BASE + path);
        } catch (e) {
            console.warn(`Failed to load ${path}:`, e.message);
            return [];
        }
    }

    // Load all data files in parallel
    const [leaderboard, runs, datasets, kdTransfer] = await Promise.all([
        loadJSON('leaderboard.json'),
        loadJSON('runs.json'),
        loadJSON('datasets.json'),
        loadJSON('kd_transfer.json'),
    ]);

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
