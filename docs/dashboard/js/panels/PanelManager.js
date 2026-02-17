/* PanelManager: reads panelConfig, builds nav + panels + controls, lazy-loads data, renders */

import { PANELS } from './panelConfig.js';
import * as Registry from '../core/Registry.js';

const BASE = 'data/';
const EXPECTED_MAJOR_VERSION = '1';

export class PanelManager {
    constructor() {
        this._charts = new Map();
        this._dataCache = new Map();
        this._metricCatalog = [];
        this._runs = [];
        this._datasets = [];
    }

    async init() {
        // Load shared reference data
        const [metricCatalog, runs, datasets] = await Promise.all([
            this._loadJSON('metrics/metric_catalog.json'),
            this._loadJSON('runs.json'),
            this._loadJSON('datasets.json'),
        ]);
        this._metricCatalog = metricCatalog;
        this._runs = runs;
        this._datasets = datasets;

        this._buildNav();
        this._buildPanels();

        const initial = this._panelFromHash() || PANELS[0].id;
        this._showPanel(initial);

        window.addEventListener('hashchange', () => {
            const id = this._panelFromHash();
            if (id) this._showPanel(id);
        });
    }

    _panelFromHash() {
        const hash = window.location.hash.replace(/^#/, '');
        if (!hash) return null;
        const panel = PANELS.find(p => p.id === hash);
        return panel ? panel.id : null;
    }

    _buildNav() {
        const nav = document.querySelector('nav');
        nav.innerHTML = '';
        PANELS.forEach((panel) => {
            const a = document.createElement('a');
            a.href = `#${panel.id}`;
            a.textContent = panel.title;
            a.addEventListener('click', (e) => {
                e.preventDefault();
                window.location.hash = panel.id;
            });
            nav.appendChild(a);
        });
    }

    _buildPanels() {
        const main = document.querySelector('main');
        main.innerHTML = '';

        PANELS.forEach(panel => {
            const section = document.createElement('section');
            section.id = panel.id;
            section.className = 'panel';
            section.style.display = 'none';

            const h2 = document.createElement('h2');
            h2.textContent = panel.title;
            section.appendChild(h2);

            if (panel.description) {
                const p = document.createElement('p');
                p.className = 'description';
                p.textContent = panel.description;
                section.appendChild(p);
            }

            if (panel.controls?.length) {
                const controls = document.createElement('div');
                controls.className = 'controls';
                panel.controls.forEach(ctrl => {
                    const label = document.createElement('label');
                    label.textContent = ctrl.label + ': ';
                    const select = document.createElement('select');
                    select.id = ctrl.id;

                    this._populateSelect(select, ctrl);

                    select.addEventListener('change', () => this._onControlChange(panel));
                    label.appendChild(select);
                    controls.appendChild(label);
                });
                section.appendChild(controls);
            }

            const chartDiv = document.createElement('div');
            chartDiv.id = `${panel.id}-chart`;
            section.appendChild(chartDiv);

            main.appendChild(section);
        });
    }

    _populateSelect(select, ctrl) {
        select.innerHTML = '';

        if (ctrl.catalogSource) {
            const metrics = this._metricCatalog.length > 0 ? this._metricCatalog : ctrl.fallback;
            metrics.forEach(m => {
                const opt = document.createElement('option');
                opt.value = m;
                opt.textContent = m.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
                if (m === ctrl.default) opt.selected = true;
                select.appendChild(opt);
            });
        } else if (ctrl.runSource) {
            const stages = ctrl.filterStages || [];
            const filtered = stages.length > 0
                ? this._runs.filter(r => r.status === 'complete' && stages.includes(r.stage))
                : this._runs.filter(r => r.status === 'complete');
            filtered.forEach(r => {
                const opt = document.createElement('option');
                opt.value = r.run_id;
                opt.textContent = r.run_id;
                select.appendChild(opt);
            });
        } else if (ctrl.datasetSource) {
            const opt0 = document.createElement('option');
            opt0.value = '';
            opt0.textContent = 'All';
            select.appendChild(opt0);
            const names = [...new Set(this._datasets.map(d => d.name))];
            names.forEach(n => {
                const opt = document.createElement('option');
                opt.value = n;
                opt.textContent = n;
                select.appendChild(opt);
            });
        } else if (ctrl.evalRunSource) {
            this._runs.filter(r => r.stage === 'evaluation' && r.status === 'complete').forEach(r => {
                const opt = document.createElement('option');
                opt.value = r.run_id;
                opt.textContent = r.run_id;
                select.appendChild(opt);
            });
        } else if (ctrl.reconErrorSource || ctrl.attentionSource || ctrl.ckaSource) {
            // Load index.json for recon_errors, attention, or cka
            const dir = ctrl.reconErrorSource ? 'recon_errors'
                : ctrl.attentionSource ? 'attention' : 'cka';
            this._loadJSON(`${dir}/index.json`).then(index => {
                select.innerHTML = '';
                const files = Array.isArray(index) ? index : (index?.data ?? []);
                if (files.length === 0) {
                    const opt = document.createElement('option');
                    opt.value = '';
                    opt.textContent = 'No data available';
                    select.appendChild(opt);
                    return;
                }
                files.forEach(fname => {
                    const opt = document.createElement('option');
                    const base = fname.replace(/\.json$/, '');
                    opt.value = base;
                    opt.textContent = base;
                    select.appendChild(opt);
                });
            }).catch(() => {
                select.innerHTML = '';
                const opt = document.createElement('option');
                opt.value = '';
                opt.textContent = 'No data available';
                select.appendChild(opt);
            });
        } else if (ctrl.embeddingSource || ctrl.dqnPolicySource) {
            // Load index.json to discover available files
            const dir = ctrl.embeddingSource ? 'embeddings' : 'dqn_policy';
            this._loadJSON(`${dir}/index.json`).then(index => {
                select.innerHTML = '';
                const files = Array.isArray(index) ? index : (index?.data ?? []);
                if (files.length === 0) {
                    const opt = document.createElement('option');
                    opt.value = '';
                    opt.textContent = 'No data available';
                    select.appendChild(opt);
                    return;
                }
                files.forEach(fname => {
                    const opt = document.createElement('option');
                    // Extract run ID from filename (strip extension and method suffix for embeddings)
                    const base = fname.replace(/\.json$/, '');
                    opt.value = base;
                    opt.textContent = base;
                    select.appendChild(opt);
                });
            }).catch(() => {
                select.innerHTML = '';
                const opt = document.createElement('option');
                opt.value = '';
                opt.textContent = 'No data available';
                select.appendChild(opt);
            });
        } else if (ctrl.options?.length) {
            ctrl.options.forEach(o => {
                const opt = document.createElement('option');
                if (typeof o === 'object') {
                    opt.value = o.value;
                    opt.textContent = o.label;
                } else {
                    opt.value = o;
                    opt.textContent = o;
                }
                if ((typeof o === 'object' ? o.value : o) === ctrl.default) opt.selected = true;
                select.appendChild(opt);
            });
        }
    }

    async _showPanel(panelId) {
        document.querySelectorAll('main > section').forEach(s => {
            s.style.display = s.id === panelId ? '' : 'none';
        });

        // Sync nav active state
        const nav = document.querySelector('nav');
        nav.querySelectorAll('a').forEach(a => {
            a.classList.toggle('active', a.getAttribute('href') === `#${panelId}`);
        });

        // Sync hash (no-op if already correct)
        if (window.location.hash !== `#${panelId}`) {
            history.replaceState(null, '', `#${panelId}`);
        }

        const panel = PANELS.find(p => p.id === panelId);
        if (!panel) return;

        // Initialize chart if not yet created
        if (!this._charts.has(panelId)) {
            await this._initChart(panel);
        }
    }

    async _initChart(panel) {
        const container = `#${panel.id}-chart`;

        // Show loading skeleton while data loads
        const containerEl = document.querySelector(container);
        if (containerEl) {
            containerEl.innerHTML = '<div class="loading-skeleton"></div>';
        }

        const ChartClass = Registry.get(panel.chartType);
        const chart = new ChartClass(container, panel.chartConfig || {});
        this._charts.set(panel.id, chart);

        let data = await this._loadPanelData(panel);
        if (data !== null) {
            const options = this._gatherOptions(panel);
            this._mergeDataOptions(data, options);
            chart.update(data, options);
        }
    }

    async _loadPanelData(panel) {
        if (panel.dynamicLoader) {
            return this._loadDynamic(panel);
        }

        if (Array.isArray(panel.dataSource)) {
            // Multi-source join
            const sources = await Promise.all(
                panel.dataSource.map(src => this._loadCached(src))
            );
            if (panel.joinFn) {
                return panel.joinFn(...sources);
            }
            return sources[0];
        }

        if (panel.dataSource) {
            let data = await this._loadCached(panel.dataSource);
            if (panel.preFilter) {
                data = panel.preFilter(data);
            }
            return data;
        }

        return null;
    }

    async _loadDynamic(panel) {
        const options = this._gatherOptions(panel);

        if (panel.dynamicLoader === 'training_curves') {
            const runId = options._run;
            if (!runId) return null;
            const fname = runId.replace(/\//g, '_') + '.json';
            return this._loadCached('training_curves/' + fname);
        }

        if (panel.dynamicLoader === 'embeddings_vgae' || panel.dynamicLoader === 'embeddings_gat') {
            const runId = options._run;
            const method = options._method || 'umap';
            if (!runId) return null;
            const model = panel.dynamicLoader.split('_')[1];
            const fname = `${runId.replace(/\//g, '_')}_${model}_${method}.json`;
            return this._loadCached('embeddings/' + fname);
        }

        if (panel.dynamicLoader === 'dqn_policy') {
            const runId = options._run;
            if (!runId) return null;
            const fname = runId.replace(/\//g, '_') + '.json';
            return this._loadCached('dqn_policy/' + fname);
        }

        if (panel.dynamicLoader === 'predictions') {
            const runId = options._run;
            if (!runId) return null;
            const fname = runId.replace(/\//g, '_') + '.json';
            const metrics = await this._loadCached('metrics/' + fname);
            if (!metrics || metrics.length === 0) return null;
            // Transform per-run metrics into bar-chart-friendly format
            const scenarioMetrics = metrics.filter(m => m.scenario !== 'val');
            return scenarioMetrics.map(m => ({
                dataset: m.scenario,
                model_type: m.model,
                scale: '',
                metric_name: m.metric_name,
                best_value: m.value,
            }));
        }

        if (panel.dynamicLoader === 'confusion_matrix') {
            const runId = options._run;
            const model = options._model || 'gat';
            const scenario = options._scenario || 'val';
            if (!runId) return null;
            const fname = runId.replace(/\//g, '_') + '.json';
            const metrics = await this._loadCached('metrics/' + fname);
            if (!metrics || metrics.length === 0) return null;
            // Find confusion_matrix metric for this model+scenario
            const cmEntry = metrics.find(m =>
                m.model === model && m.scenario === scenario && m.metric_name === 'confusion_matrix'
            );
            if (!cmEntry || !cmEntry.value) return null;
            const cm = typeof cmEntry.value === 'string' ? JSON.parse(cmEntry.value) : cmEntry.value;
            return {
                matrix: cm,
                rowLabels: ['Actual Normal', 'Actual Attack'],
                colLabels: ['Pred Normal', 'Pred Attack'],
            };
        }

        if (panel.dynamicLoader === 'roc_curves') {
            const runId = options._run;
            const curveType = options._curveType || 'roc';
            if (!runId) return null;
            const runSlug = runId.replace(/\//g, '_');
            // Load all model curves for this run
            const index = await this._loadCached('roc_curves/index.json');
            const files = Array.isArray(index) ? index : (index?.data ?? []);
            const matching = files.filter(f => f.startsWith(runSlug));
            if (matching.length === 0) return null;
            const series = [];
            for (const fname of matching) {
                const curveData = await this._loadCached('roc_curves/' + fname);
                if (!curveData) continue;
                const d = curveData.data || curveData;
                const model = d.model || fname.split('_').pop().replace('.json', '');
                if (curveType === 'roc' && d.roc_curve) {
                    series.push({
                        name: model.toUpperCase(),
                        auc: d.auc,
                        points: d.roc_curve.fpr.map((fpr, i) => ({ x: fpr, y: d.roc_curve.tpr[i] })),
                    });
                } else if (curveType === 'pr' && d.pr_curve) {
                    series.push({
                        name: model.toUpperCase(),
                        auc: d.pr_auc,
                        points: d.pr_curve.recall.map((rec, i) => ({ x: rec, y: d.pr_curve.precision[i] })),
                    });
                }
            }
            return {
                series,
                _curveType: curveType,
                _xLabel: curveType === 'roc' ? 'False Positive Rate' : 'Recall',
                _yLabel: curveType === 'roc' ? 'True Positive Rate' : 'Precision',
                _refLine: curveType === 'roc' ? 'diagonal' : 'none',
            };
        }

        if (panel.dynamicLoader === 'recon_errors') {
            const runId = options._run;
            if (!runId) return null;
            const fname = runId.replace(/\//g, '_') + '.json';
            const recon = await this._loadCached('recon_errors/' + fname);
            if (!recon) return null;
            const d = recon.data || recon;
            // Transform to histogram-friendly format (alpha_by_label style)
            const normalErrors = [], attackErrors = [];
            for (let i = 0; i < d.errors.length; i++) {
                if (d.labels[i] === 0) normalErrors.push(d.errors[i]);
                else attackErrors.push(d.errors[i]);
            }
            return {
                alpha_by_label: { normal: normalErrors, attack: attackErrors },
                _thresholdLine: d.optimal_threshold,
                _xLabel: 'Reconstruction Error',
            };
        }

        if (panel.dynamicLoader === 'attention') {
            const runId = options._run;
            const layer = parseInt(options._layer || '0', 10);
            const sampleIdx = parseInt(options._sample || '0', 10);
            if (!runId) return null;
            const fname = runId.replace(/\//g, '_') + '.json';
            const attnData = await this._loadCached('attention/' + fname);
            if (!attnData || attnData.length === 0) return null;
            const samples = Array.isArray(attnData) ? attnData : (attnData.data || []);
            const sample = samples[sampleIdx] || samples[0];
            if (!sample) return null;
            // Build force graph data with attention edge weights
            const edgeIndex = sample.edge_index;
            const nodeFeatures = sample.node_features;
            const layerData = sample.layers[layer] || sample.layers[0];
            const numNodes = nodeFeatures.length;
            const nodes = nodeFeatures.map((feat, i) => ({ id: i, features: [feat] }));
            const links = [];
            const edgeWeights = [];
            for (let i = 0; i < edgeIndex[0].length; i++) {
                links.push({ source: edgeIndex[0][i], target: edgeIndex[1][i] });
                edgeWeights.push(layerData.alpha_mean[i] || 0);
            }
            return [{
                dataset: 'attention',
                sample_idx: sampleIdx,
                label: sample.label,
                nodes,
                links,
                _edgeWeights: edgeWeights,
            }];
        }

        if (panel.dynamicLoader === 'attention_carpet') {
            const runId = options._run;
            const sampleIdx = parseInt(options._sample || '0', 10);
            if (!runId) return null;
            const fname = runId.replace(/\//g, '_') + '.json';
            const attnData = await this._loadCached('attention/' + fname);
            if (!attnData || attnData.length === 0) return null;
            const samples = Array.isArray(attnData) ? attnData : (attnData.data || []);
            const sample = samples[sampleIdx] || samples[0];
            if (!sample || !sample.layers) return null;
            // Build carpet: rows=edges, cols=layers, value=mean attention
            const nEdges = Math.min(sample.layers[0].alpha_mean.length, 30); // cap for readability
            const nLayers = sample.layers.length;
            const matrix = [];
            for (let e = 0; e < nEdges; e++) {
                const row = [];
                for (let l = 0; l < nLayers; l++) {
                    row.push(sample.layers[l].alpha_mean[e] || 0);
                }
                matrix.push(row);
            }
            return {
                matrix,
                rowLabels: Array.from({ length: nEdges }, (_, i) => `E${i}`),
                colLabels: sample.layers.map((_, i) => `Layer ${i + 1}`),
            };
        }

        if (panel.dynamicLoader === 'training_carpet') {
            const metric = options._metric || 'val_loss';
            // Load all training curve files
            const index = await this._loadCached('training_curves/index.json').catch(() => []);
            const runs = this._runs.filter(r =>
                r.status === 'complete' && ['autoencoder', 'curriculum', 'fusion'].includes(r.stage)
            );
            if (runs.length === 0) return null;
            const matrix = [];
            const rowLabels = [];
            let maxEpochs = 0;
            for (const run of runs.slice(0, 20)) { // cap at 20 runs
                const fname = run.run_id.replace(/\//g, '_') + '.json';
                try {
                    const curveData = await this._loadCached('training_curves/' + fname);
                    if (!curveData || curveData.length === 0) continue;
                    const filtered = curveData.filter(d => d.metric_name === metric);
                    if (filtered.length === 0) continue;
                    const sorted = filtered.sort((a, b) => a.epoch - b.epoch);
                    const row = sorted.map(d => d.value);
                    matrix.push(row);
                    rowLabels.push(run.run_id.split('/').pop() || run.run_id);
                    maxEpochs = Math.max(maxEpochs, row.length);
                } catch { continue; }
            }
            if (matrix.length === 0) return null;
            // Pad rows to same length
            for (const row of matrix) {
                while (row.length < maxEpochs) row.push(null);
            }
            return {
                matrix,
                rowLabels,
                colLabels: Array.from({ length: maxEpochs }, (_, i) => `${i + 1}`),
            };
        }

        if (panel.dynamicLoader === 'cka') {
            const runId = options._run;
            if (!runId) return null;
            const fname = runId.replace(/\//g, '_') + '.json';
            const ckaData = await this._loadCached('cka/' + fname);
            if (!ckaData) return null;
            const d = ckaData.data || ckaData;
            return {
                matrix: d.matrix,
                rowLabels: d.teacher_layers || d.matrix.map((_, i) => `Teacher L${i + 1}`),
                colLabels: d.student_layers || d.matrix[0].map((_, i) => `Student L${i + 1}`),
            };
        }

        return null;
    }

    async _loadCached(path) {
        if (this._dataCache.has(path)) {
            return this._dataCache.get(path);
        }
        const data = await this._loadJSON(path);
        this._dataCache.set(path, data);
        return data;
    }

    async _loadJSON(path) {
        try {
            const raw = await d3.json(BASE + path);
            if (!raw) return [];
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

    _gatherOptions(panel) {
        const options = { ...(panel.chartConfig || {}) };
        (panel.controls || []).forEach(ctrl => {
            const el = document.getElementById(ctrl.id);
            if (el) {
                let val = el.value;
                // Type coercion for numbers
                if (ctrl.mapTo === 'bins') val = parseInt(val, 10);
                if (ctrl.mapTo === 'label' && val !== '') val = parseInt(val, 10);
                if (ctrl.mapTo === '_dataset_filter' && val) {
                    // Special: filter data by dataset
                    options._dataset_filter = val;
                }
                options[ctrl.mapTo] = val;
            }
        });
        return options;
    }

    async _onControlChange(panel) {
        const chart = this._charts.get(panel.id);
        if (!chart) return;

        // For dynamic loaders, need to reload data
        const needsReload = (panel.controls || []).some(
            ctrl => ctrl.mapTo?.startsWith('_') && ctrl.mapTo !== '_dataset_filter'
        );

        let data;
        if (panel.dynamicLoader || needsReload) {
            // Clear cache for dynamic data
            this._dataCache.delete(panel.id + '_dynamic');
            data = await this._loadPanelData(panel);
        } else {
            data = await this._loadPanelData(panel);
        }

        const options = this._gatherOptions(panel);

        // Apply dataset filter if present
        if (options._dataset_filter && Array.isArray(data)) {
            data = data.filter(d => d.dataset === options._dataset_filter);
        }

        if (data !== null) {
            this._mergeDataOptions(data, options);
            chart.update(data, options);
        }
    }

    _mergeDataOptions(data, options) {
        // Dynamic loaders can embed options in the data object via _-prefixed keys
        if (data && typeof data === 'object' && !Array.isArray(data)) {
            for (const key of Object.keys(data)) {
                if (key.startsWith('_')) {
                    const optKey = key.slice(1);
                    if (!(optKey in options) || options[optKey] == null) {
                        options[optKey] = data[key];
                    }
                }
            }
        }
        // For arrays (e.g., force graph attention data), check first element
        if (Array.isArray(data) && data.length > 0 && data[0]) {
            for (const key of Object.keys(data[0])) {
                if (key.startsWith('_')) {
                    const optKey = key.slice(1);
                    if (!(optKey in options) || options[optKey] == null) {
                        options[optKey] = data[0][key];
                    }
                }
            }
        }
    }
}
