/* Histogram chart: binned distribution (e.g., DQN alpha values) */

import { BaseChart } from '../core/BaseChart.js';
import { LABEL_COLORS, COLORS } from '../core/Theme.js';
import { register } from '../core/Registry.js';

export class HistogramChart extends BaseChart {
    render(data, options = {}) {
        const g = this._g;
        const w = this._width, h = this._height;
        const bins = options.bins || 20;

        // Expect data = { alphas: [], labels: [], alpha_by_label: {normal:[], attack:[]} }
        const normalAlphas = data.alpha_by_label?.normal || [];
        const attackAlphas = data.alpha_by_label?.attack || [];

        if (normalAlphas.length === 0 && attackAlphas.length === 0) {
            return this._showNoData('No distribution data');
        }

        // Dynamic domain: default [0,1] for alphas, compute from data for errors
        const allValues = [...normalAlphas, ...attackAlphas];
        const extent = d3.extent(allValues);
        const domain = (extent[0] >= 0 && extent[1] <= 1.01)
            ? [0, 1]
            : [extent[0] * 0.95, extent[1] * 1.05];

        const x = d3.scaleLinear().domain(domain).range([0, w]);

        const histogram = d3.bin().domain(domain).thresholds(bins);
        const normalBins = histogram(normalAlphas);
        const attackBins = histogram(attackAlphas);

        const maxCount = d3.max([...normalBins, ...attackBins], b => b.length);
        const y = d3.scaleLinear().domain([0, maxCount]).nice().range([h, 0]);

        this._addXAxis(x);
        this._addYAxis(y);
        this._addGrid(y);
        this._addAxisLabel(options.xLabel || 'Alpha (VGAE weight)', 'x');
        this._addAxisLabel('Count', 'y');

        // Normal bars
        g.selectAll('.bar-normal')
            .data(normalBins)
            .join('rect')
            .attr('class', 'bar-normal')
            .attr('x', d => x(d.x0) + 1)
            .attr('y', d => y(d.length))
            .attr('width', d => Math.max(0, x(d.x1) - x(d.x0) - 2))
            .attr('height', d => h - y(d.length))
            .attr('fill', LABEL_COLORS.normal)
            .attr('opacity', 0.6)
            .on('mouseover', (event, d) => {
                this._showTooltip(`Normal: ${d.length}<br>Range: ${d.x0.toFixed(2)}–${d.x1.toFixed(2)}`, event);
            })
            .on('mousemove', (event) => this._moveTooltip(event))
            .on('mouseout', () => this._hideTooltip());

        // Attack bars (stacked on top)
        g.selectAll('.bar-attack')
            .data(attackBins)
            .join('rect')
            .attr('class', 'bar-attack')
            .attr('x', d => x(d.x0) + 1)
            .attr('y', d => y(d.length))
            .attr('width', d => Math.max(0, x(d.x1) - x(d.x0) - 2))
            .attr('height', d => h - y(d.length))
            .attr('fill', LABEL_COLORS.attack)
            .attr('opacity', 0.6)
            .on('mouseover', (event, d) => {
                this._showTooltip(`Attack: ${d.length}<br>Range: ${d.x0.toFixed(2)}–${d.x1.toFixed(2)}`, event);
            })
            .on('mousemove', (event) => this._moveTooltip(event))
            .on('mouseout', () => this._hideTooltip());

        // Threshold line (opt-in)
        const thresholdLine = options.thresholdLine;
        if (thresholdLine != null && isFinite(thresholdLine)) {
            const tx = x(thresholdLine);
            g.append('line')
                .attr('x1', tx).attr('y1', 0)
                .attr('x2', tx).attr('y2', h)
                .attr('stroke', '#d29922')
                .attr('stroke-width', 2)
                .attr('stroke-dasharray', '6,3');
            g.append('text')
                .attr('x', tx + 4).attr('y', 12)
                .attr('fill', '#d29922')
                .style('font-size', '10px')
                .text(`θ=${thresholdLine.toFixed(4)}`);
        }

        this._addLegend([
            { label: 'Normal', color: LABEL_COLORS.normal },
            { label: 'Attack', color: LABEL_COLORS.attack },
        ], w - 100, 0);
    }
}

register('histogram', HistogramChart);
