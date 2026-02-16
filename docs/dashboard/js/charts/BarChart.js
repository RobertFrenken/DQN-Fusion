/* Grouped bar chart: metric across datasets per model config */

import { BaseChart } from '../core/BaseChart.js';
import { COLORS } from '../core/Theme.js';
import { register } from '../core/Registry.js';

export class BarChart extends BaseChart {
    render(data, options = {}) {
        const metric = options.metric || 'f1';
        const filtered = data.filter(d => d.metric_name === metric);
        if (filtered.length === 0) return this._showNoData();

        const g = this._g;
        const w = this._width, h = this._height;

        const datasets = [...new Set(filtered.map(d => d.dataset))];
        const configs = [...new Set(filtered.map(d => `${d.model_type}_${d.scale}`))];

        const x0 = d3.scaleBand().domain(datasets).range([0, w]).padding(0.2);
        const x1 = d3.scaleBand().domain(configs).range([0, x0.bandwidth()]).padding(0.05);
        const y = d3.scaleLinear().domain([0, 1]).range([h, 0]);
        const color = d3.scaleOrdinal().domain(configs).range(COLORS);

        this._addXAxis(x0);
        this._addYAxis(y, 5);
        this._addGrid(y);

        datasets.forEach(ds => {
            const dsData = filtered.filter(d => d.dataset === ds);
            configs.forEach(cfg => {
                const d = dsData.find(d => `${d.model_type}_${d.scale}` === cfg);
                if (!d || d.best_value == null) return;
                g.append('rect')
                    .attr('x', x0(ds) + x1(cfg))
                    .attr('y', y(d.best_value))
                    .attr('width', x1.bandwidth())
                    .attr('height', h - y(d.best_value))
                    .attr('fill', color(cfg))
                    .attr('rx', 2)
                    .on('mouseover', (event) => {
                        this._showTooltip(`${cfg}<br>${ds}: ${d.best_value.toFixed(4)}`, event);
                    })
                    .on('mousemove', (event) => this._moveTooltip(event))
                    .on('mouseout', () => this._hideTooltip());
            });
        });

        this._addLegend(
            configs.map(c => ({ label: c, color: color(c) })),
            w - 120, 0
        );
    }
}

register('bar', BarChart);
