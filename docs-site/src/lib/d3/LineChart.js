/* Line chart: training curves (epoch vs metric), supports multi-series */

import * as d3 from 'd3';
import { BaseChart } from './BaseChart.js';
import { COLORS, METRIC_COLORS, MODEL_COLORS, semanticColorScale } from './Theme.js';

export class LineChart extends BaseChart {
    render(data, options = {}) {
        // Support multi-series: options.metrics (array) or options.metric (string, may be comma-separated)
        let metrics;
        if (options.metrics) {
            metrics = options.metrics;
        } else {
            const raw = options.metric || 'val_loss';
            metrics = raw.includes(',') ? raw.split(',').map(s => s.trim()) : [raw];
        }

        const allFiltered = data.filter(d => metrics.includes(d.metric_name));
        if (allFiltered.length === 0) return this._showNoData(`No data for metric: ${metrics.join(', ')}`);

        const g = this._g;
        const w = this._width, h = this._height;

        const x = d3.scaleLinear()
            .domain(d3.extent(allFiltered, d => d.epoch))
            .range([0, w]);
        const y = d3.scaleLinear()
            .domain(d3.extent(allFiltered, d => d.value))
            .nice()
            .range([h, 0]);

        this._addXAxis(x, 10);
        this._addYAxis(y);
        this._addGrid(y);
        this._addAxisLabel('Epoch', 'x');
        this._addAxisLabel(metrics.length === 1 ? metrics[0] : 'Value', 'y');

        const line = d3.line()
            .defined(d => d.value != null)
            .x(d => x(d.epoch))
            .y(d => y(d.value));

        const colorScale = semanticColorScale(metrics, METRIC_COLORS);

        metrics.forEach(metric => {
            const series = allFiltered.filter(d => d.metric_name === metric)
                .sort((a, b) => a.epoch - b.epoch);
            if (series.length === 0) return;

            g.append('path')
                .datum(series)
                .attr('fill', 'none')
                .attr('stroke', colorScale(metric))
                .attr('stroke-width', 2)
                .attr('d', line);
        });

        if (metrics.length > 1) {
            this._addLegend(
                metrics.map(m => ({ label: m.replace(/_/g, ' '), color: colorScale(m) })),
                w - 130, 0
            );
        }
    }

    /**
     * Render multiple runs overlaid on the same axes.
     * data format: { runs: [{ run_id, points: [{epoch, metric_name, value}] }] }
     */
    renderMultiRun(data, options = {}) {
        if (!data || !data.runs || data.runs.length === 0) return this._showNoData();

        const metric = options.metric || 'val_loss';
        const metrics = metric.includes(',') ? metric.split(',').map(s => s.trim()) : [metric];
        const g = this._g;
        const w = this._width, h = this._height;

        // Collect all filtered points across runs
        const allPoints = [];
        const seriesData = [];
        data.runs.forEach((run, ri) => {
            metrics.forEach(m => {
                const pts = run.points
                    .filter(p => p.metric_name === m)
                    .sort((a, b) => a.epoch - b.epoch);
                if (pts.length === 0) return;
                allPoints.push(...pts);
                const label = data.runs.length > 1
                    ? `${(run.run_id || '').split('/').pop()} · ${m}`
                    : m;
                seriesData.push({ label, points: pts, runIdx: ri, metric: m });
            });
        });

        if (allPoints.length === 0) return this._showNoData(`No data for metric: ${metrics.join(', ')}`);

        const x = d3.scaleLinear()
            .domain(d3.extent(allPoints, d => d.epoch))
            .range([0, w]);
        const y = d3.scaleLinear()
            .domain(d3.extent(allPoints, d => d.value))
            .nice()
            .range([h, 0]);

        this._addXAxis(x, 10);
        this._addYAxis(y);
        this._addGrid(y);
        this._addAxisLabel('Epoch', 'x');
        this._addAxisLabel(metrics.length === 1 ? metrics[0] : 'Value', 'y');

        const line = d3.line()
            .defined(d => d.value != null)
            .x(d => x(d.epoch))
            .y(d => y(d.value));

        // Color per series: alternate between run colors and metric colors
        const runColors = [MODEL_COLORS.vgae, MODEL_COLORS.gat, MODEL_COLORS.dqn, '#f85149'];
        const seriesColors = seriesData.map((s, i) => {
            if (data.runs.length === 1) return METRIC_COLORS[s.metric] || COLORS[i % COLORS.length];
            // Multi-run: use run index for base color, dash pattern for metric
            return runColors[s.runIdx % runColors.length];
        });

        seriesData.forEach((s, i) => {
            const dashArray = (data.runs.length > 1 && metrics.indexOf(s.metric) > 0) ? '6,3' : 'none';
            g.append('path')
                .datum(s.points)
                .attr('fill', 'none')
                .attr('stroke', seriesColors[i])
                .attr('stroke-width', 2)
                .attr('stroke-dasharray', dashArray)
                .attr('d', line);
        });

        this._addLegend(
            seriesData.map((s, i) => ({ label: s.label, color: seriesColors[i] })),
            w - 180, 0
        );
    }
}
