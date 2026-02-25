/* Bubble chart: extends ScatterChart with size encoding for 3-metric comparison */

import { ScatterChart } from './ScatterChart.js';

export class BubbleChart extends ScatterChart {
    render(data, options = {}) {
        const xField = options.xField || 'accuracy';
        const yField = options.yField || 'f1';
        const bubbleOpts = {
            sizeField: options.sizeField || 'param_count',
            xField,
            yField,
            colorField: options.colorField || 'model_type',
            xLabel: options.xLabel || 'Accuracy',
            yLabel: options.yLabel || 'F1 Score',
            xDomain: options.xDomain || this._computeDomain(data, xField, { padding: 0.05, min: 0, max: 1 }),
            yDomain: options.yDomain || this._computeDomain(data, yField, { padding: 0.05, min: 0, max: 1 }),
            tooltipFn: d => `${d.model_type} (${d.scale})<br>${d.dataset || ''}<br>` +
                `Acc: ${(d.accuracy ?? 0).toFixed(4)}<br>F1: ${(d.f1 ?? 0).toFixed(4)}<br>` +
                `Params: ${d.param_count_M ? d.param_count_M + 'M' : 'N/A'}`,
            ...options,
        };
        // Don't let explicit options override computed domains if they were hardcoded [0,1]
        if (!options.xDomain) bubbleOpts.xDomain = this._computeDomain(data, xField, { padding: 0.05, min: 0, max: 1 });
        if (!options.yDomain) bubbleOpts.yDomain = this._computeDomain(data, yField, { padding: 0.05, min: 0, max: 1 });
        super.render(data, bubbleOpts);
    }
}
