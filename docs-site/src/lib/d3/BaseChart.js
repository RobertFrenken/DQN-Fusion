/* Base chart class: SVG setup, margins, tooltip, responsive, state management */

import * as d3 from 'd3';
import { MARGIN } from './Theme.js';

export class BaseChart {
    constructor(container, config = {}) {
        this.container = d3.select(container);
        this.config = config;
        this.margin = config.margin || MARGIN;
        this._tooltip = null;
        this._svg = null;
        this._g = null;
        this._resizeObserver = null;
        this.init();
    }

    init() {
        this._setupSVG();
        this._setupTooltip();
        this._setupResize();
    }

    _setupSVG() {
        this.container.html('');
        const rect = this.container.node().getBoundingClientRect();
        this._width = (rect.width || 600) - this.margin.left - this.margin.right;
        this._height = (this.config.height || 350) - this.margin.top - this.margin.bottom;

        this._svg = this.container.append('svg')
            .attr('width', this._width + this.margin.left + this.margin.right)
            .attr('height', this._height + this.margin.top + this.margin.bottom);

        this._g = this._svg.append('g')
            .attr('transform', `translate(${this.margin.left},${this.margin.top})`);
    }

    _setupTooltip() {
        this._tooltip = d3.select('body').append('div').attr('class', 'tooltip');
    }

    _setupResize() {
        if (typeof ResizeObserver === 'undefined') return;
        this._resizeObserver = new ResizeObserver(() => {
            if (this._lastData !== undefined) {
                this.update(this._lastData, this._lastOptions);
            }
        });
        this._resizeObserver.observe(this.container.node());
    }

    update(data, options = {}) {
        this._lastData = data;
        this._lastOptions = options;
        this._setupSVG();

        if (!data || (Array.isArray(data) && data.length === 0)) {
            this._showNoData(options.noDataMessage);
            return;
        }
        try {
            this.render(data, options);
        } catch (e) {
            console.error(`${this.constructor.name} render error:`, e);
            this._showError('Render error');
        }
    }

    /** Subclasses implement this */
    render(data, options) {
        throw new Error('Subclass must implement render()');
    }

    destroy() {
        if (this._resizeObserver) {
            this._resizeObserver.disconnect();
        }
        if (this._tooltip) {
            this._tooltip.remove();
        }
        this.container.html('');
    }

    // --- Shared infrastructure ---

    _showTooltip(html, event) {
        this._tooltip.style('opacity', 1).html(html);
        if (event) {
            this._tooltip
                .style('left', (event.pageX + 10) + 'px')
                .style('top', (event.pageY - 20) + 'px');
        }
    }

    _moveTooltip(event) {
        this._tooltip
            .style('left', (event.pageX + 10) + 'px')
            .style('top', (event.pageY - 20) + 'px');
    }

    _hideTooltip() {
        this._tooltip.style('opacity', 0);
    }

    _showNoData(message) {
        this.container.html(`<div class="no-data">${message || 'No data available'}</div>`);
    }

    _showLoading() {
        this.container.html('<div class="no-data">Loading...</div>');
    }

    _showError(message) {
        this.container.html(`<div class="no-data">${message}</div>`);
    }

    _addXAxis(scale, ticks) {
        this._g.append('g')
            .attr('class', 'axis')
            .attr('transform', `translate(0,${this._height})`)
            .call(d3.axisBottom(scale).ticks(ticks || null));
    }

    _addYAxis(scale, ticks) {
        this._g.append('g')
            .attr('class', 'axis')
            .call(d3.axisLeft(scale).ticks(ticks || null));
    }

    _addGrid(yScale, ticks = 5) {
        this._g.append('g')
            .attr('class', 'grid')
            .call(d3.axisLeft(yScale).ticks(ticks).tickSize(-this._width).tickFormat(''));
    }

    _addAxisLabel(text, axis = 'x') {
        if (axis === 'x') {
            this._g.append('text')
                .attr('x', this._width / 2).attr('y', this._height + 40)
                .attr('text-anchor', 'middle').attr('fill', '#8b949e').style('font-size', '12px')
                .text(text);
        } else {
            this._g.append('text')
                .attr('transform', 'rotate(-90)').attr('x', -this._height / 2).attr('y', -45)
                .attr('text-anchor', 'middle').attr('fill', '#8b949e').style('font-size', '12px')
                .text(text);
        }
    }

    _computeDomain(data, field, opts = {}) {
        const { padding = 0.05, min: absMin, max: absMax } = opts;
        const values = data.map(d => d[field]).filter(v => v != null && isFinite(v));
        if (values.length === 0) return [0, 1];
        let [lo, hi] = d3.extent(values);
        if (lo === hi) { lo -= 0.5; hi += 0.5; }
        const span = hi - lo;
        lo -= span * padding;
        hi += span * padding;
        if (absMin != null) lo = Math.max(lo, absMin);
        if (absMax != null) hi = Math.min(hi, absMax);
        return [lo, hi];
    }

    _addLegend(items, x, y) {
        const legend = this._g.append('g').attr('transform', `translate(${x}, ${y})`);
        items.forEach(({ label, color }, i) => {
            const g = legend.append('g').attr('transform', `translate(0, ${i * 18})`);
            g.append('rect').attr('width', 12).attr('height', 12).attr('fill', color).attr('rx', 2);
            g.append('text').attr('x', 16).attr('y', 10).attr('fill', '#c9d1d9').style('font-size', '11px').text(label);
        });
    }

    get width() { return this._width; }
    get height() { return this._height; }
    get g() { return this._g; }
    get svg() { return this._svg; }
}
