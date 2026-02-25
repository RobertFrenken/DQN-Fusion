/**
 * Force-directed graph: CAN bus graph structure visualization.
 * Extracted from docs-site/src/lib/d3/ForceGraph.js — standalone function, no class/BaseChart.
 *
 * Usage in OJS cell:
 *   import { renderForceGraph } from "./_ojs/force-graph.js"
 *   renderForceGraph(d3, container, data, { dataset: "hcrl_ch", colorBy: "can_id" })
 */

import { COLORS, LABEL_COLORS } from "./theme.js";

const FEATURE_NAMES = [
  'CAN_ID', 'Data_0', 'Data_1', 'Data_2', 'Data_3',
  'Data_4', 'Data_5', 'Data_6', 'Data_7', 'Count', 'Position',
];

/**
 * Render a force-directed CAN bus graph into a container element.
 * @param {object} d3 - D3 module (import * as d3 from "d3")
 * @param {HTMLElement} container - DOM element to render into
 * @param {Array} data - Array of graph samples [{dataset, label, nodes, links}, ...]
 * @param {object} options - {dataset, label, colorBy, edgeWeights, width, height}
 * @returns {{ simulation: object, destroy: function }} cleanup handle
 */
export function renderForceGraph(d3, container, data, options = {}) {
  const dataset = options.dataset || null;
  const labelFilter = options.label;
  const colorBy = options.colorBy || 'can_id';
  const width = options.width || 680;
  const height = options.height || 500;

  let samples = data;
  if (dataset) samples = samples.filter(d => d.dataset === dataset);
  if (labelFilter !== undefined && labelFilter !== null) {
    samples = samples.filter(d => d.label === labelFilter);
  }

  if (samples.length === 0) {
    container.innerHTML = '<p style="color:#6b7280;text-align:center;padding:2rem;font-style:italic">No graph samples for selection</p>';
    return { simulation: null, destroy() {} };
  }

  const sample = samples[0];
  const nodes = sample.nodes.map(n => ({ ...n }));
  const links = sample.links.map(l => ({ source: l.source, target: l.target }));

  // Clear container
  container.innerHTML = '';
  const svg = d3.select(container)
    .append('svg')
    .attr('width', width)
    .attr('height', height)
    .attr('viewBox', [0, 0, width, height]);

  const g = svg.append('g');

  // Degree for size encoding
  const degree = new Map();
  links.forEach(l => {
    degree.set(l.source, (degree.get(l.source) || 0) + 1);
    degree.set(l.target, (degree.get(l.target) || 0) + 1);
  });

  const maxDeg = Math.max(...degree.values(), 1);
  const rScale = d3.scaleSqrt().domain([0, maxDeg]).range([3, 12]);

  // Color function
  let colorFn;
  if (colorBy === 'degree') {
    const degreeColor = d3.scaleSequential(d3.interpolateViridis).domain([0, maxDeg]);
    colorFn = d => degreeColor(degree.get(d.id) || 0);
  } else {
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
    .force('center', d3.forceCenter(width / 2, height / 2))
    .force('collision', d3.forceCollide().radius(d => rScale(degree.get(d.id) || 0) + 2));

  // Edge attention weights (optional)
  const edgeWeights = options.edgeWeights || null;
  let edgeWidthFn = () => 1;
  let edgeColorFn = () => '#d1d5db';
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

  // Tooltip
  const tooltip = d3.select(container)
    .append('div')
    .attr('class', 'tooltip');

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
      tooltip
        .style('opacity', 1)
        .html(buildTooltip(d, degree));
    })
    .on('mousemove', (event) => {
      tooltip
        .style('left', (event.offsetX + 12) + 'px')
        .style('top', (event.offsetY - 10) + 'px');
    })
    .on('mouseout', () => {
      tooltip.style('opacity', 0);
    });

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
      return s === d.id || t === d.id ? '#2563eb' : '#d1d5db';
    });
  });

  svg.on('click', () => {
    node.attr('opacity', 1.0);
    link.attr('stroke-opacity', 0.4).attr('stroke', '#d1d5db');
  });

  // Label
  const labelText = sample.label === 1 ? 'Attack' : sample.label === 0 ? 'Normal' : 'Unknown';
  g.append('text')
    .attr('x', 5).attr('y', 15)
    .attr('fill', sample.label === 1 ? LABEL_COLORS.attack : LABEL_COLORS.normal)
    .style('font-size', '12px').style('font-weight', '600')
    .text(`${sample.dataset} \u2014 ${labelText} (${nodes.length} nodes, ${links.length} edges)`);

  simulation.on('tick', () => {
    link
      .attr('x1', d => d.source.x).attr('y1', d => d.source.y)
      .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
    node
      .attr('cx', d => Math.max(5, Math.min(width - 5, d.x)))
      .attr('cy', d => Math.max(5, Math.min(height - 5, d.y)));
  });

  return {
    simulation,
    destroy() {
      simulation.stop();
      container.innerHTML = '';
    }
  };
}

function buildTooltip(d, degree) {
  let html = `<strong>Node ${d.id}</strong><br>Degree: ${degree.get(d.id) || 0}`;
  if (d.features && d.features.length > 0) {
    html += '<br><hr style="margin:4px 0;border-color:#e2e8f0">';
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
