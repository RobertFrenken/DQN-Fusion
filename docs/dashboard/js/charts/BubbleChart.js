/* Bubble chart: extends ScatterChart with size encoding for 3-metric comparison */

import { ScatterChart } from './ScatterChart.js';
import { register } from '../core/Registry.js';

export class BubbleChart extends ScatterChart {
    render(data, options = {}) {
        // Default bubble options
        const bubbleOpts = {
            sizeField: options.sizeField || 'param_count',
            xField: options.xField || 'accuracy',
            yField: options.yField || 'f1',
            colorField: options.colorField || 'model_type',
            xLabel: options.xLabel || 'Accuracy',
            yLabel: options.yLabel || 'F1 Score',
            xDomain: options.xDomain || [0, 1],
            yDomain: options.yDomain || [0, 1],
            tooltipFn: d => `${d.model_type} (${d.scale})<br>${d.dataset || ''}<br>` +
                `Acc: ${(d.accuracy ?? 0).toFixed(4)}<br>F1: ${(d.f1 ?? 0).toFixed(4)}<br>` +
                `Params: ${d.param_count_M ? d.param_count_M + 'M' : 'N/A'}`,
            ...options,
        };
        super.render(data, bubbleOpts);
    }
}

register('bubble', BubbleChart);
