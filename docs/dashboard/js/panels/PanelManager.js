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
        this._showPanel(PANELS[0].id);
    }

    _buildNav() {
        const nav = document.querySelector('nav');
        nav.innerHTML = '';
        PANELS.forEach((panel, i) => {
            const a = document.createElement('a');
            a.href = `#${panel.id}`;
            a.textContent = panel.title;
            if (i === 0) a.classList.add('active');
            a.addEventListener('click', (e) => {
                e.preventDefault();
                nav.querySelectorAll('a').forEach(l => l.classList.remove('active'));
                a.classList.add('active');
                this._showPanel(panel.id);
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
        } else if (ctrl.embeddingSource || ctrl.dqnPolicySource) {
            // Will be populated after data is loaded
            const opt = document.createElement('option');
            opt.value = '';
            opt.textContent = 'Loading...';
            select.appendChild(opt);
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

        const panel = PANELS.find(p => p.id === panelId);
        if (!panel) return;

        // Initialize chart if not yet created
        if (!this._charts.has(panelId)) {
            await this._initChart(panel);
        }
    }

    async _initChart(panel) {
        const container = `#${panel.id}-chart`;
        const ChartClass = Registry.get(panel.chartType);
        const chart = new ChartClass(container, panel.chartConfig || {});
        this._charts.set(panel.id, chart);

        const data = await this._loadPanelData(panel);
        if (data !== null) {
            const options = this._gatherOptions(panel);
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
            chart.update(data, options);
        }
    }
}
