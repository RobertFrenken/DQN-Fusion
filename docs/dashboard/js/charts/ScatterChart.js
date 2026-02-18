/* Scatter chart: generic scatter plot with configurable axes */

import { BaseChart } from '../core/BaseChart.js';
import { COLORS, LABEL_COLORS, DATASET_COLORS, MODEL_COLORS, CONFIG_COLORS, semanticColorScale } from '../core/Theme.js';
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
        const showDensity = options.showDensity === true || options.showDensity === 'density';

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
            // Pick semantic map based on colorField
            const semanticMap = colorField === 'dataset' ? DATASET_COLORS
                : colorField === 'config' ? CONFIG_COLORS
                : colorField === 'model_type' ? MODEL_COLORS
                : null;
            const scale = semanticMap
                ? semanticColorScale(colorDomain, semanticMap)
                : d3.scaleOrdinal().domain(colorDomain).range(COLORS);
            colorFn = d => scale(d[colorField]);
        }

        const radiusFn = sizeField
            ? d3.scaleSqrt().domain(d3.extent(data, d => d[sizeField])).range([3, 15])
            : null;

        // Density contour overlay (opt-in)
        if (showDensity && data.length > 10 && typeof d3.contourDensity === 'function') {
            const density = d3.contourDensity()
                .x(d => x(d[xField]))
                .y(d => y(d[yField]))
                .size([w, h])
                .bandwidth(20)
                .thresholds(8)(data);

            g.append('g')
                .attr('class', 'density-contours')
                .selectAll('path')
                .data(density)
                .join('path')
                .attr('d', d3.geoPath())
                .attr('fill', 'none')
                .attr('stroke', '#58a6ff')
                .attr('stroke-opacity', (d, i) => 0.15 + i * 0.02)
                .attr('stroke-width', 1);
        }

        // Voronoi overlay (opt-in)
        const showVoronoi = options.showDensity === 'voronoi';
        if (showVoronoi && data.length > 2 && typeof d3.Delaunay !== 'undefined') {
            const points = data.map(d => [x(d[xField]), y(d[yField])]);
            const delaunay = d3.Delaunay.from(points);
            const voronoi = delaunay.voronoi([0, 0, w, h]);

            g.append('g')
                .attr('class', 'voronoi-overlay')
                .selectAll('path')
                .data(data)
                .join('path')
                .attr('d', (d, i) => voronoi.renderCell(i))
                .attr('fill', d => colorFn(d))
                .attr('fill-opacity', 0.06)
                .attr('stroke', d => colorFn(d))
                .attr('stroke-opacity', 0.15)
                .attr('stroke-width', 0.5);
        }

        const pointOpacity = (showDensity && !showVoronoi) ? 0.4 : showVoronoi ? 0.9 : 0.8;
        const pointRadius = (showDensity && !showVoronoi) ? 3 : showVoronoi ? 4 : 6;

        g.selectAll('circle')
            .data(data)
            .join('circle')
            .attr('cx', d => x(d[xField]))
            .attr('cy', d => y(d[yField]))
            .attr('r', d => radiusFn ? radiusFn(d[sizeField]) : pointRadius)
            .attr('fill', colorFn)
            .attr('opacity', pointOpacity)
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

        // Pareto frontier overlay
        if (options.paretoLine) {
            const valid = data.filter(d => d[xField] != null && d[yField] != null);
            const sorted = [...valid].sort((a, b) => a[xField] - b[xField]);
            // Sweep right-to-left to find non-dominated points (maximize y)
            const frontier = [];
            let maxY = -Infinity;
            for (let i = sorted.length - 1; i >= 0; i--) {
                if (sorted[i][yField] >= maxY) {
                    maxY = sorted[i][yField];
                    frontier.unshift(sorted[i]);
                }
            }
            if (frontier.length > 1) {
                const line = d3.line()
                    .x(d => x(d[xField]))
                    .y(d => y(d[yField]))
                    .curve(d3.curveStepAfter);
                g.append('path')
                    .datum(frontier)
                    .attr('d', line)
                    .attr('fill', 'none')
                    .attr('stroke', '#f0883e')
                    .attr('stroke-width', 2)
                    .attr('stroke-dasharray', '6,3')
                    .attr('opacity', 0.8);
            }
        }
    }
}

register('scatter', ScatterChart);
