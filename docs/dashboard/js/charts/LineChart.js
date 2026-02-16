/* Line chart: training curves (epoch vs metric), supports multi-series */

import { BaseChart } from '../core/BaseChart.js';
import { COLORS } from '../core/Theme.js';
import { register } from '../core/Registry.js';

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

        const colorScale = d3.scaleOrdinal().domain(metrics).range(COLORS);

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
}

register('line', LineChart);
