/* Timeline scatter: run history (x=date, y=dataset, color=status) */

import * as d3 from 'd3';
import { BaseChart } from './BaseChart.js';
import { STATUS_COLORS } from './Theme.js';

export class TimelineChart extends BaseChart {
    render(data, options = {}) {
        const withDates = data.filter(d => d.started_at);
        if (withDates.length === 0) return this._showNoData('No runs with timestamps');

        const g = this._g;
        const w = this._width;
        const datasets = [...new Set(withDates.map(d => d.dataset))];
        const h = Math.max(200, datasets.length * 30) - this.margin.top - this.margin.bottom;

        // Resize SVG for dynamic height
        this._svg
            .attr('height', h + this.margin.top + this.margin.bottom);

        withDates.forEach(d => {
            d._date = new Date(d.started_at);
        });

        const x = d3.scaleTime()
            .domain(d3.extent(withDates, d => d._date))
            .range([0, w]);
        const y = d3.scaleBand()
            .domain(datasets)
            .range([0, h])
            .padding(0.3);

        g.append('g').attr('class', 'axis').attr('transform', `translate(0,${h})`)
            .call(d3.axisBottom(x).ticks(6));
        g.append('g').attr('class', 'axis').call(d3.axisLeft(y));

        g.selectAll('circle')
            .data(withDates)
            .join('circle')
            .attr('cx', d => x(d._date))
            .attr('cy', d => y(d.dataset) + y.bandwidth() / 2)
            .attr('r', 5)
            .attr('fill', d => STATUS_COLORS[d.status] || STATUS_COLORS.unknown)
            .attr('opacity', 0.8)
            .on('mouseover', (event, d) => {
                this._showTooltip(`${d.run_id}<br>${d.stage} / ${d.status}<br>${d.started_at}`, event);
            })
            .on('mousemove', (event) => this._moveTooltip(event))
            .on('mouseout', () => this._hideTooltip());
    }
}
