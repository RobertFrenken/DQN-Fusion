/** Chart types available in D3Chart.svelte registry. */
export type ChartType =
  | 'scatter'
  | 'bar'
  | 'line'
  | 'histogram'
  | 'heatmap'
  | 'curve'
  | 'force'
  | 'timeline'
  | 'bubble'
  | 'table'
  | 'summary';

/** Layout width classes for the CSS Grid system. */
export type LayoutWidth = 'body' | 'wide' | 'full';

/** Which renderer a figure uses. */
export type Renderer = 'plot' | 'd3';

/** Observable Plot mark config — passed to PlotFigure.svelte. */
export interface PlotMarkConfig {
  type: string;
  data?: unknown[];
  [key: string]: unknown;
}
