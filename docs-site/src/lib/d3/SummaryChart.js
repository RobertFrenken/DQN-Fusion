/* Summary overview: at-a-glance metric cards */

import * as d3 from 'd3';
import { BaseChart } from './BaseChart.js';

export class SummaryChart extends BaseChart {
    _setupSVG() {
        this.container.html('');
        this._width = 0;
        this._height = 0;
    }

    render(data, options = {}) {
        const { leaderboard = [], sizes = [], kd = [] } = data;
        if (leaderboard.length === 0) return this._showNoData();

        const f1Rows = leaderboard.filter(d => d.metric_name === 'f1');

        // Best F1
        const best = f1Rows.reduce((a, b) =>
            (b.best_value ?? 0) > (a.best_value ?? 0) ? b : a, f1Rows[0]);

        // Datasets with runs
        const datasetCount = new Set(leaderboard.map(d => d.dataset)).size;

        // Completed runs (distinct model identifiers)
        const runCount = new Set(leaderboard.map(d => d.model)).size;

        // KD gap: average (teacher - student) F1
        const kdF1 = kd.filter(d => d.metric_name === 'f1' && d.teacher_value != null && d.student_value != null);
        const kdGap = kdF1.length > 0
            ? kdF1.reduce((s, d) => s + (d.teacher_value - d.student_value), 0) / kdF1.length
            : null;

        // Best model type by average F1
        const byType = new Map();
        f1Rows.forEach(d => {
            if (d.best_value == null) return;
            if (!byType.has(d.model_type)) byType.set(d.model_type, []);
            byType.get(d.model_type).push(d.best_value);
        });
        let bestType = null, bestTypeAvg = -1;
        byType.forEach((vals, type) => {
            const avg = vals.reduce((a, b) => a + b, 0) / vals.length;
            if (avg > bestTypeAvg) { bestTypeAvg = avg; bestType = type; }
        });

        // Compression ratio: large vs small params
        const largeParams = sizes.filter(s => s.scale === 'large');
        const smallParams = sizes.filter(s => s.scale === 'small');
        let compression = null;
        if (largeParams.length > 0 && smallParams.length > 0) {
            const avgLarge = largeParams.reduce((s, d) => s + (d.param_count || 0), 0) / largeParams.length;
            const avgSmall = smallParams.reduce((s, d) => s + (d.param_count || 0), 0) / smallParams.length;
            if (avgSmall > 0) compression = avgLarge / avgSmall;
        }

        const cards = [
            {
                value: best ? (best.best_value ?? 0).toFixed(4) : '--',
                label: 'Best F1',
                subtitle: best ? `${best.model_type}/${best.dataset}` : '',
            },
            {
                value: datasetCount,
                label: 'Datasets',
                subtitle: 'with runs',
            },
            {
                value: runCount,
                label: 'Completed Runs',
                subtitle: 'distinct configs',
            },
            {
                value: kdGap != null ? kdGap.toFixed(4) : '--',
                label: 'KD Gap',
                subtitle: kdGap != null ? 'avg teacher \u2212 student F1' : 'no KD data',
            },
            {
                value: bestType ? bestType.toUpperCase() : '--',
                label: 'Best Model',
                subtitle: bestType ? `avg F1: ${bestTypeAvg.toFixed(4)}` : '',
            },
            {
                value: compression ? `${compression.toFixed(1)}x` : '--',
                label: 'Compression',
                subtitle: compression ? 'large / small params' : 'no size data',
            },
        ];

        const grid = this.container.append('div').attr('class', 'summary-grid');
        cards.forEach(c => {
            const card = grid.append('div').attr('class', 'summary-card');
            card.append('div').attr('class', 'card-value').text(c.value);
            card.append('div').attr('class', 'card-label').text(c.label);
            if (c.subtitle) {
                card.append('div').attr('class', 'card-subtitle').text(c.subtitle);
            }
        });
    }
}
