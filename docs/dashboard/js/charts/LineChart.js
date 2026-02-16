/* Line chart: training curves (epoch vs metric) */

import { BaseChart } from '../core/BaseChart.js';
import { COLORS } from '../core/Theme.js';
import { register } from '../core/Registry.js';

export class LineChart extends BaseChart {
    render(data, options = {}) {
        const metric = options.metric || 'val_loss';
        const filtered = data.filter(d => d.metric_name === metric);
        if (filtered.length === 0) return this._showNoData(`No data for metric: ${metric}`);

        const g = this._g;
        const w = this._width, h = this._height;

        const x = d3.scaleLinear()
            .domain(d3.extent(filtered, d => d.epoch))
            .range([0, w]);
        const y = d3.scaleLinear()
            .domain(d3.extent(filtered, d => d.value))
            .nice()
            .range([h, 0]);

        this._addXAxis(x, 10);
        this._addYAxis(y);
        this._addGrid(y);
        this._addAxisLabel('Epoch', 'x');
        this._addAxisLabel(metric, 'y');

        const line = d3.line()
            .defined(d => d.value != null)
            .x(d => x(d.epoch))
            .y(d => y(d.value));

        g.append('path')
            .datum(filtered)
            .attr('fill', 'none')
            .attr('stroke', COLORS[0])
            .attr('stroke-width', 2)
            .attr('d', line);
    }
}

register('line', LineChart);
