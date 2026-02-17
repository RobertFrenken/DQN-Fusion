/* Heatmap chart: confusion matrix, carpet plot, CKA matrix */

import { BaseChart } from '../core/BaseChart.js';
import { LABEL_COLORS } from '../core/Theme.js';
import { register } from '../core/Registry.js';

export class HeatmapChart extends BaseChart {
    render(data, options = {}) {
        const g = this._g;
        const w = this._width, h = this._height;

        // data = { matrix: [[...], ...], rowLabels: [...], colLabels: [...] }
        const matrix = data.matrix;
        if (!matrix || matrix.length === 0) return this._showNoData('No matrix data');

        const rowLabels = data.rowLabels || matrix.map((_, i) => `${i}`);
        const colLabels = data.colLabels || (matrix[0] || []).map((_, i) => `${i}`);
        const nRows = matrix.length;
        const nCols = (matrix[0] || []).length;

        const colorScheme = options.colorScale || 'sequential';
        const format = options.format || 'float';
        const showAnnotations = options.annotations !== false;
        const title = options.title || '';

        // Flatten values for domain
        const allValues = matrix.flat();
        const [minVal, maxVal] = d3.extent(allValues);

        let colorFn;
        if (colorScheme === 'diverging') {
            const absMax = Math.max(Math.abs(minVal), Math.abs(maxVal));
            colorFn = d3.scaleSequential(d3.interpolateRdBu).domain([absMax, -absMax]);
        } else {
            colorFn = d3.scaleSequential(d3.interpolateBlues).domain([minVal, maxVal]);
        }

        // Cell sizing
        const cellW = Math.min(w / nCols, 80);
        const cellH = Math.min(h / nRows, 60);
        const gridW = cellW * nCols;
        const gridH = cellH * nRows;
        const offsetX = (w - gridW) / 2;
        const offsetY = (h - gridH) / 2 + 10;

        // Title
        if (title) {
            g.append('text')
                .attr('x', w / 2).attr('y', offsetY - 15)
                .attr('text-anchor', 'middle')
                .attr('fill', '#c9d1d9').style('font-size', '13px').style('font-weight', '600')
                .text(title);
        }

        // Column labels (top)
        g.selectAll('.col-label')
            .data(colLabels)
            .join('text')
            .attr('class', 'col-label')
            .attr('x', (d, i) => offsetX + i * cellW + cellW / 2)
            .attr('y', offsetY - 4)
            .attr('text-anchor', 'middle')
            .attr('fill', '#8b949e').style('font-size', '11px')
            .text(d => d);

        // Row labels (left)
        g.selectAll('.row-label')
            .data(rowLabels)
            .join('text')
            .attr('class', 'row-label')
            .attr('x', offsetX - 6)
            .attr('y', (d, i) => offsetY + i * cellH + cellH / 2 + 4)
            .attr('text-anchor', 'end')
            .attr('fill', '#8b949e').style('font-size', '11px')
            .text(d => d);

        // Cells
        const cells = [];
        for (let r = 0; r < nRows; r++) {
            for (let c = 0; c < nCols; c++) {
                cells.push({ row: r, col: c, value: matrix[r][c] });
            }
        }

        const total = allValues.reduce((s, v) => s + v, 0);

        g.selectAll('.heatmap-cell')
            .data(cells)
            .join('rect')
            .attr('class', 'heatmap-cell')
            .attr('x', d => offsetX + d.col * cellW)
            .attr('y', d => offsetY + d.row * cellH)
            .attr('width', cellW - 1)
            .attr('height', cellH - 1)
            .attr('fill', d => colorFn(d.value))
            .attr('rx', 3)
            .on('mouseover', (event, d) => {
                const pct = total > 0 ? (d.value / total * 100).toFixed(1) : '0.0';
                this._showTooltip(
                    `${rowLabels[d.row]} / ${colLabels[d.col]}<br>Value: ${this._formatValue(d.value, format)}<br>${pct}%`,
                    event
                );
            })
            .on('mousemove', (event) => this._moveTooltip(event))
            .on('mouseout', () => this._hideTooltip());

        // Text annotations
        if (showAnnotations) {
            g.selectAll('.heatmap-text')
                .data(cells)
                .join('text')
                .attr('class', 'heatmap-text')
                .attr('x', d => offsetX + d.col * cellW + cellW / 2)
                .attr('y', d => offsetY + d.row * cellH + cellH / 2)
                .attr('text-anchor', 'middle')
                .attr('dominant-baseline', 'central')
                .attr('fill', d => this._textColor(d.value, minVal, maxVal))
                .style('font-size', nCols <= 4 ? '13px' : '10px')
                .style('font-weight', '600')
                .style('font-family', 'var(--font-mono)')
                .text(d => this._formatValue(d.value, format));

            // Percentage below count for confusion matrix format
            if (format === 'count' && total > 0) {
                g.selectAll('.heatmap-pct')
                    .data(cells)
                    .join('text')
                    .attr('class', 'heatmap-pct')
                    .attr('x', d => offsetX + d.col * cellW + cellW / 2)
                    .attr('y', d => offsetY + d.row * cellH + cellH / 2 + 14)
                    .attr('text-anchor', 'middle')
                    .attr('fill', d => this._textColor(d.value, minVal, maxVal))
                    .style('font-size', '10px')
                    .style('opacity', 0.7)
                    .text(d => `${(d.value / total * 100).toFixed(1)}%`);
            }
        }
    }

    _formatValue(val, format) {
        if (format === 'count') return val.toLocaleString();
        if (format === 'pct') return (val * 100).toFixed(1) + '%';
        return typeof val === 'number' ? val.toFixed(3) : val;
    }

    _textColor(val, minVal, maxVal) {
        // Use white text on dark cells, dark text on light cells
        const range = maxVal - minVal || 1;
        const normalized = (val - minVal) / range;
        return normalized > 0.6 ? '#0d1117' : '#c9d1d9';
    }
}

register('heatmap', HeatmapChart);
