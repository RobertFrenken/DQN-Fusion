/* Scatter chart: generic scatter plot with configurable axes */

import { BaseChart } from '../core/BaseChart.js';
import { COLORS, LABEL_COLORS } from '../core/Theme.js';
import { register } from '../core/Registry.js';

export class ScatterChart extends BaseChart {
    render(data, options = {}) {
        const g = this._g;
        const w = this._width, h = this._height;
        const xField = options.xField || 'teacher_value';
        const yField = options.yField || 'student_value';
        const colorField = options.colorField || 'dataset';
        const xLabel = options.xLabel || xField;
        const yLabel = options.yLabel || yField;
        const xDomain = options.xDomain || d3.extent(data, d => d[xField]);
        const yDomain = options.yDomain || d3.extent(data, d => d[yField]);
        const sizeField = options.sizeField || null;

        const x = d3.scaleLinear().domain(xDomain).range([0, w]).nice();
        const y = d3.scaleLinear().domain(yDomain).range([h, 0]).nice();

        this._addXAxis(x);
        this._addYAxis(y);
        this._addAxisLabel(xLabel, 'x');
        this._addAxisLabel(yLabel, 'y');

        // Reference line for diagonal comparisons
        if (options.diagonalLine) {
            g.append('line')
                .attr('x1', x(xDomain[0])).attr('y1', y(xDomain[0]))
                .attr('x2', x(xDomain[1])).attr('y2', y(xDomain[1]))
                .attr('stroke', '#30363d').attr('stroke-dasharray', '4,4');
        }

        const colorDomain = [...new Set(data.map(d => d[colorField]))];
        let colorFn;
        if (options.labelColors) {
            colorFn = d => {
                const label = d[colorField];
                return label === 0 || label === 'normal' ? LABEL_COLORS.normal : LABEL_COLORS.attack;
            };
        } else {
            const scale = d3.scaleOrdinal().domain(colorDomain).range(COLORS);
            colorFn = d => scale(d[colorField]);
        }

        const radiusFn = sizeField
            ? d3.scaleSqrt().domain(d3.extent(data, d => d[sizeField])).range([3, 15])
            : null;

        g.selectAll('circle')
            .data(data)
            .join('circle')
            .attr('cx', d => x(d[xField]))
            .attr('cy', d => y(d[yField]))
            .attr('r', d => radiusFn ? radiusFn(d[sizeField]) : 6)
            .attr('fill', colorFn)
            .attr('opacity', 0.8)
            .attr('stroke', '#fff')
            .attr('stroke-width', 1)
            .on('mouseover', (event, d) => {
                const tip = options.tooltipFn
                    ? options.tooltipFn(d)
                    : `${d[colorField]}<br>${xLabel}: ${(d[xField] ?? 0).toFixed(4)}<br>${yLabel}: ${(d[yField] ?? 0).toFixed(4)}`;
                this._showTooltip(tip, event);
            })
            .on('mousemove', (event) => this._moveTooltip(event))
            .on('mouseout', () => this._hideTooltip());
    }
}

register('scatter', ScatterChart);
