/* Force-directed graph: CAN bus graph structure visualization */

import { BaseChart } from '../core/BaseChart.js';
import { COLORS, LABEL_COLORS } from '../core/Theme.js';
import { register } from '../core/Registry.js';

const FEATURE_NAMES = [
    'CAN_ID', 'Data_0', 'Data_1', 'Data_2', 'Data_3',
    'Data_4', 'Data_5', 'Data_6', 'Data_7', 'Count', 'Position',
];

export class ForceGraph extends BaseChart {
    render(data, options = {}) {
        const dataset = options.dataset || null;
        const labelFilter = options.label;
        const colorBy = options.colorBy || 'can_id';

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

        // Color by CAN_ID or degree
        let colorFn;
        if (colorBy === 'degree') {
            const degreeColor = d3.scaleSequential(d3.interpolateViridis).domain([0, maxDeg]);
            colorFn = d => degreeColor(degree.get(d.id) || 0);
        } else {
            // CAN_ID-based coloring
            const canIds = [...new Set(nodes.map(n => {
                const f = n.features?.[0];
                return f != null ? Math.floor(f) : n.id;
            }))].sort((a, b) => a - b);
            let canColor;
            if (canIds.length <= 20) {
                canColor = d3.scaleOrdinal().domain(canIds).range(COLORS);
            } else {
                const rankMap = new Map(canIds.map((id, i) => [id, i]));
                const seqScale = d3.scaleSequential(d3.interpolateTurbo).domain([0, canIds.length - 1]);
                canColor = id => seqScale(rankMap.get(id) ?? 0);
            }
            colorFn = d => {
                const f = d.features?.[0];
                const canId = f != null ? Math.floor(f) : d.id;
                return typeof canColor === 'function' ? canColor(canId) : canColor(canId);
            };
        }

        const simulation = d3.forceSimulation(nodes)
            .force('link', d3.forceLink(links).id(d => d.id).distance(30))
            .force('charge', d3.forceManyBody().strength(-40))
            .force('center', d3.forceCenter(w / 2, h / 2))
            .force('collision', d3.forceCollide().radius(d => rScale(degree.get(d.id) || 0) + 2));

        // Edge attention weights (opt-in via options.edgeWeights)
        const edgeWeights = options.edgeWeights || null;
        let edgeWidthFn = () => 1;
        let edgeColorFn = () => '#30363d';
        let edgeOpacityFn = () => 0.4;

        if (edgeWeights && edgeWeights.length === links.length) {
            const wExtent = d3.extent(edgeWeights);
            const widthScale = d3.scaleLinear().domain(wExtent).range([0.5, 4]);
            const colorInterp = d3.scaleSequential(d3.interpolateOrRd).domain(wExtent);
            edgeWidthFn = (d, i) => widthScale(edgeWeights[i]);
            edgeColorFn = (d, i) => colorInterp(edgeWeights[i]);
            edgeOpacityFn = (d, i) => 0.3 + 0.6 * ((edgeWeights[i] - wExtent[0]) / (wExtent[1] - wExtent[0] || 1));
        }

        const link = g.append('g')
            .attr('class', 'force-links')
            .selectAll('line')
            .data(links)
            .join('line')
            .attr('stroke', edgeColorFn)
            .attr('stroke-opacity', edgeOpacityFn)
            .attr('stroke-width', edgeWidthFn);

        const node = g.append('g')
            .attr('class', 'force-nodes')
            .selectAll('circle')
            .data(nodes)
            .join('circle')
            .attr('r', d => rScale(degree.get(d.id) || 0))
            .attr('fill', colorFn)
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
                this._showTooltip(this._buildTooltip(d, degree), event);
            })
            .on('mousemove', (event) => this._moveTooltip(event))
            .on('mouseout', () => this._hideTooltip());

        // Click-to-highlight neighborhood
        const neighborSet = new Map();
        links.forEach(l => {
            const s = typeof l.source === 'object' ? l.source.id : l.source;
            const t = typeof l.target === 'object' ? l.target.id : l.target;
            if (!neighborSet.has(s)) neighborSet.set(s, new Set());
            if (!neighborSet.has(t)) neighborSet.set(t, new Set());
            neighborSet.get(s).add(t);
            neighborSet.get(t).add(s);
        });

        node.on('click', (event, d) => {
            event.stopPropagation();
            const neighbors = neighborSet.get(d.id) || new Set();
            node.attr('opacity', n => n.id === d.id || neighbors.has(n.id) ? 1.0 : 0.1);
            link.attr('stroke-opacity', l => {
                const s = typeof l.source === 'object' ? l.source.id : l.source;
                const t = typeof l.target === 'object' ? l.target.id : l.target;
                return s === d.id || t === d.id ? 0.8 : 0.05;
            }).attr('stroke', l => {
                const s = typeof l.source === 'object' ? l.source.id : l.source;
                const t = typeof l.target === 'object' ? l.target.id : l.target;
                return s === d.id || t === d.id ? '#58a6ff' : '#30363d';
            });
        });

        this._svg.on('click', () => {
            node.attr('opacity', 1.0);
            link.attr('stroke-opacity', 0.4).attr('stroke', '#30363d');
        });

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

        this._simulation = simulation;
    }

    _buildTooltip(d, degree) {
        let html = `<strong>Node ${d.id}</strong><br>Degree: ${degree.get(d.id) || 0}`;
        if (d.features && d.features.length > 0) {
            html += '<br><hr style="margin:4px 0;border-color:#30363d">';
            d.features.forEach((val, i) => {
                const name = FEATURE_NAMES[i] || `Feature_${i}`;
                let display;
                if (name === 'CAN_ID') {
                    display = '0x' + Math.floor(val).toString(16).toUpperCase().padStart(3, '0');
                } else {
                    display = Number.isInteger(val) ? val : val.toFixed(4);
                }
                html += `<br>${name}: ${display}`;
            });
        }
        return html;
    }

    destroy() {
        if (this._simulation) this._simulation.stop();
        super.destroy();
    }
}

register('force', ForceGraph);
