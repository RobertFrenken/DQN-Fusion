/* Curve chart: ROC and PR curves with reference lines and AUC annotation */

import { BaseChart } from '../core/BaseChart.js';
import { COLORS } from '../core/Theme.js';
import { register } from '../core/Registry.js';

export class CurveChart extends BaseChart {
    render(data, options = {}) {
        const g = this._g;
        const w = this._width, h = this._height;

        // data = { series: [{ name, points: [{x, y}], auc }] }
        const series = data.series;
        if (!series || series.length === 0) return this._showNoData('No curve data');

        const xLabel = options.xLabel || data._xLabel || 'X';
        const yLabel = options.yLabel || data._yLabel || 'Y';
        const xDomain = options.xDomain || [0, 1];
        const yDomain = options.yDomain || [0, 1];
        const refLine = options.refLine || data._refLine || 'none';

        const x = d3.scaleLinear().domain(xDomain).range([0, w]);
        const y = d3.scaleLinear().domain(yDomain).range([h, 0]);

        this._addXAxis(x);
        this._addYAxis(y);
        this._addGrid(y);
        this._addAxisLabel(xLabel, 'x');
        this._addAxisLabel(yLabel, 'y');

        // Reference line (diagonal for ROC random classifier)
        if (refLine === 'diagonal') {
            g.append('line')
                .attr('x1', x(xDomain[0])).attr('y1', y(yDomain[0]))
                .attr('x2', x(xDomain[1])).attr('y2', y(yDomain[1]))
                .attr('stroke', '#30363d')
                .attr('stroke-dasharray', '6,4')
                .attr('stroke-width', 1);
        }

        const colorScale = d3.scaleOrdinal()
            .domain(series.map(s => s.name))
            .range(COLORS);

        const line = d3.line()
            .x(d => x(d.x))
            .y(d => y(d.y))
            .curve(d3.curveLinear);

        // Draw each series
        series.forEach(s => {
            if (!s.points || s.points.length === 0) return;

            g.append('path')
                .datum(s.points)
                .attr('d', line)
                .attr('fill', 'none')
                .attr('stroke', colorScale(s.name))
                .attr('stroke-width', 2)
                .attr('opacity', 0.9);

            // Invisible wider path for hover detection
            g.append('path')
                .datum(s.points)
                .attr('d', line)
                .attr('fill', 'none')
                .attr('stroke', 'transparent')
                .attr('stroke-width', 12)
                .on('mouseover', (event) => {
                    const aucStr = s.auc != null ? `<br>AUC: ${s.auc.toFixed(4)}` : '';
                    this._showTooltip(`<strong>${s.name}</strong>${aucStr}`, event);
                })
                .on('mousemove', (event) => this._moveTooltip(event))
                .on('mouseout', () => this._hideTooltip());
        });

        // Legend with AUC values
        const legendItems = series.map(s => ({
            label: s.auc != null ? `${s.name} (AUC=${s.auc.toFixed(3)})` : s.name,
            color: colorScale(s.name),
        }));
        this._addLegend(legendItems, 10, 5);
    }
}

register('curve', CurveChart);
