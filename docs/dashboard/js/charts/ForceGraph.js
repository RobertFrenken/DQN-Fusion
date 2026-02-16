/* Force-directed graph: CAN bus graph structure visualization */

import { BaseChart } from '../core/BaseChart.js';
import { COLORS, LABEL_COLORS } from '../core/Theme.js';
import { register } from '../core/Registry.js';

export class ForceGraph extends BaseChart {
    render(data, options = {}) {
        const dataset = options.dataset || null;
        const labelFilter = options.label;

        let samples = data;
        if (dataset) samples = samples.filter(d => d.dataset === dataset);
        if (labelFilter !== undefined && labelFilter !== null) {
            samples = samples.filter(d => d.label === labelFilter);
        }

        if (samples.length === 0) return this._showNoData('No graph samples for selection');

        // Use the first matching sample
        const sample = samples[0];
        const nodes = sample.nodes.map(n => ({ ...n }));
        const links = sample.links.map(l => ({
            source: l.source,
            target: l.target,
        }));

        const g = this._g;
        const w = this._width, h = this._height;

        // Degree for size encoding
        const degree = new Map();
        links.forEach(l => {
            degree.set(l.source, (degree.get(l.source) || 0) + 1);
            degree.set(l.target, (degree.get(l.target) || 0) + 1);
        });

        const maxDeg = Math.max(...degree.values(), 1);
        const rScale = d3.scaleSqrt().domain([0, maxDeg]).range([3, 12]);
        const nodeColor = d3.scaleOrdinal().range(COLORS);

        const simulation = d3.forceSimulation(nodes)
            .force('link', d3.forceLink(links).id(d => d.id).distance(30))
            .force('charge', d3.forceManyBody().strength(-40))
            .force('center', d3.forceCenter(w / 2, h / 2))
            .force('collision', d3.forceCollide().radius(d => rScale(degree.get(d.id) || 0) + 2));

        const link = g.append('g')
            .attr('class', 'force-links')
            .selectAll('line')
            .data(links)
            .join('line')
            .attr('stroke', '#30363d')
            .attr('stroke-opacity', 0.4)
            .attr('stroke-width', 1);

        const node = g.append('g')
            .attr('class', 'force-nodes')
            .selectAll('circle')
            .data(nodes)
            .join('circle')
            .attr('r', d => rScale(degree.get(d.id) || 0))
            .attr('fill', d => nodeColor(d.id % COLORS.length))
            .attr('stroke', '#fff')
            .attr('stroke-width', 0.5)
            .call(d3.drag()
                .on('start', (event, d) => {
                    if (!event.active) simulation.alphaTarget(0.3).restart();
                    d.fx = d.x; d.fy = d.y;
                })
                .on('drag', (event, d) => { d.fx = event.x; d.fy = event.y; })
                .on('end', (event, d) => {
                    if (!event.active) simulation.alphaTarget(0);
                    d.fx = null; d.fy = null;
                }))
            .on('mouseover', (event, d) => {
                this._showTooltip(
                    `Node ${d.id}<br>Degree: ${degree.get(d.id) || 0}`,
                    event
                );
            })
            .on('mousemove', (event) => this._moveTooltip(event))
            .on('mouseout', () => this._hideTooltip());

        // Label
        const labelText = sample.label === 1 ? 'Attack' : sample.label === 0 ? 'Normal' : 'Unknown';
        g.append('text')
            .attr('x', 5).attr('y', 15)
            .attr('fill', sample.label === 1 ? LABEL_COLORS.attack : LABEL_COLORS.normal)
            .style('font-size', '12px').style('font-weight', '600')
            .text(`${sample.dataset} — ${labelText} (${nodes.length} nodes, ${links.length} edges)`);

        simulation.on('tick', () => {
            link
                .attr('x1', d => d.source.x).attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
            node
                .attr('cx', d => Math.max(5, Math.min(w - 5, d.x)))
                .attr('cy', d => Math.max(5, Math.min(h - 5, d.y)));
        });

        // Store simulation for cleanup
        this._simulation = simulation;
    }

    destroy() {
        if (this._simulation) this._simulation.stop();
        super.destroy();
    }
}

register('force', ForceGraph);
