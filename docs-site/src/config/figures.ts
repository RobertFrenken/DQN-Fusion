import type { ChartType, LayoutWidth, Renderer, PlotMarkConfig } from './shared';

export interface FigureConfig {
  id: string;
  number: number;
  caption: string;
  /** Which renderer to use. Defaults to 'd3'. */
  renderer: Renderer;
  chartType: ChartType;
  /** Tier 2 runtime fetch path (relative to /data/). null = use staticData. */
  dataSource: string | null;
  /** Tier 1 content collection entry ID (for build-time data). */
  staticData?: string;
  defaultOptions: Record<string, unknown>;
  /** Observable Plot marks config (used when renderer === 'plot'). */
  plotMarks?: PlotMarkConfig[];
  /** Observable Plot top-level options (grid, color, axes, etc). */
  plotOptions?: Record<string, unknown>;
  layout: LayoutWidth;
  section: string;
}

/**
 * Paper figures. Add new figures here — one registry entry + one MDX line.
 *
 * Numbering follows the paper order. Sections correspond to MDX page slugs.
 */
export const FIGURES: FigureConfig[] = [
  {
    id: 'fig-graph-structure',
    number: 1,
    caption: 'CAN message graph structure with node degree sizing.',
    renderer: 'd3',
    chartType: 'force',
    dataSource: 'graph_samples.json',
    defaultOptions: {},
    layout: 'wide',
    section: '01-introduction',
  },
  {
    id: 'fig-recon-errors',
    number: 2,
    caption: 'VGAE reconstruction error distributions for normal vs. attack traffic.',
    renderer: 'd3',
    chartType: 'histogram',
    dataSource: null,
    defaultOptions: {},
    layout: 'body',
    section: '03-vgae',
  },
  {
    id: 'fig-training-curves',
    number: 3,
    caption: 'Training loss curves across all datasets.',
    renderer: 'd3',
    chartType: 'line',
    dataSource: null,
    defaultOptions: {},
    layout: 'wide',
    section: '04-gat',
  },
  {
    id: 'fig-attention-weights',
    number: 4,
    caption: 'GAT attention head weight distributions across message types.',
    renderer: 'd3',
    chartType: 'heatmap',
    dataSource: null,
    defaultOptions: {},
    layout: 'wide',
    section: '04-gat',
  },
  {
    id: 'fig-roc-curves',
    number: 5,
    caption: 'ROC curves comparing VGAE, GAT, and DQN-Fusion models.',
    renderer: 'd3',
    chartType: 'curve',
    dataSource: null,
    defaultOptions: {},
    layout: 'body',
    section: '05-dqn-fusion',
  },
  {
    id: 'fig-dqn-policy',
    number: 6,
    caption: 'DQN fusion policy alpha values across attack classes.',
    renderer: 'd3',
    chartType: 'bar',
    dataSource: null,
    defaultOptions: {},
    layout: 'body',
    section: '05-dqn-fusion',
  },
  {
    id: 'fig-kd-transfer',
    number: 7,
    caption: 'Teacher vs. Student F1 scores across six CAN bus datasets.',
    renderer: 'plot',
    chartType: 'scatter',
    staticData: 'kdTransfer',
    dataSource: null,
    defaultOptions: {},
    plotMarks: [
      {
        type: 'dot',
        x: 'teacher_value',
        y: 'student_value',
        fill: 'dataset',
        tip: true,
        r: 4,
      },
      {
        type: 'ruleY',
        data: [0],
        stroke: '#ccc',
        strokeDasharray: '4,4',
      },
    ],
    plotOptions: {
      grid: true,
      x: { label: 'Teacher F1 Score' },
      y: { label: 'Student F1 Score' },
      color: { legend: true },
    },
    layout: 'body',
    section: '06-knowledge-distillation',
  },
  {
    id: 'fig-embeddings',
    number: 8,
    caption: 'VGAE latent space projections (UMAP) colored by traffic label.',
    renderer: 'd3',
    chartType: 'scatter',
    dataSource: null,
    defaultOptions: {
      xField: 'dim0',
      yField: 'dim1',
      colorField: 'label',
      xLabel: 'UMAP Dimension 1',
      yLabel: 'UMAP Dimension 2',
      labelColors: true,
    },
    layout: 'wide',
    section: '07-evaluation',
  },
  {
    id: 'fig-model-sizes',
    number: 9,
    caption: 'Parameter counts: large teacher vs. small student models.',
    renderer: 'plot',
    chartType: 'bar',
    staticData: 'modelSizes',
    dataSource: null,
    defaultOptions: {},
    plotMarks: [
      {
        type: 'barY',
        x: 'model_type',
        y: 'param_count_M',
        fill: 'scale',
        tip: true,
      },
    ],
    plotOptions: {
      grid: true,
      x: { label: 'Model' },
      y: { label: 'Parameters (M)' },
      color: { legend: true },
    },
    layout: 'body',
    section: '06-knowledge-distillation',
  },
  {
    id: 'fig-cka-similarity',
    number: 10,
    caption: 'CKA similarity between teacher and student layer representations.',
    renderer: 'd3',
    chartType: 'heatmap',
    dataSource: null,
    defaultOptions: {},
    layout: 'wide',
    section: '06-knowledge-distillation',
  },
];

/** Look up a figure by ID. Throws if not found. */
export function getFigure(id: string): FigureConfig {
  const fig = FIGURES.find((f) => f.id === id);
  if (!fig) throw new Error(`Unknown figure: ${id}`);
  return fig;
}

/** Get all figures for a given paper section. */
export function getFiguresBySection(section: string): FigureConfig[] {
  return FIGURES.filter((f) => f.section === section);
}
