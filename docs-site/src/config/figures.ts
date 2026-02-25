import type { ChartType, LayoutWidth } from './shared';

export interface FigureConfig {
  id: string;
  number: number;
  caption: string;
  chartType: ChartType;
  /** Tier 2 runtime fetch path (relative to /data/). null = use staticData. */
  dataSource: string | null;
  /** Tier 1 content collection entry ID (for build-time data). */
  staticData?: string;
  defaultOptions: Record<string, unknown>;
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
    chartType: 'histogram',
    dataSource: null, // per-run, fetched by interactive figure
    defaultOptions: {},
    layout: 'body',
    section: '03-vgae',
  },
  {
    id: 'fig-training-curves',
    number: 3,
    caption: 'Training loss curves across all datasets.',
    chartType: 'line',
    dataSource: null, // per-run, fetched by interactive figure
    defaultOptions: {},
    layout: 'wide',
    section: '04-gat',
  },
  {
    id: 'fig-attention-weights',
    number: 4,
    caption: 'GAT attention head weight distributions across message types.',
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
    chartType: 'scatter',
    staticData: 'kdTransfer',
    dataSource: null,
    defaultOptions: {
      xField: 'teacher_value',
      yField: 'student_value',
      colorField: 'dataset',
      xLabel: 'Teacher F1 Score',
      yLabel: 'Student F1 Score',
      diagonalLine: true,
    },
    layout: 'body',
    section: '06-knowledge-distillation',
  },
  {
    id: 'fig-embeddings',
    number: 8,
    caption: 'VGAE latent space projections (UMAP) colored by traffic label.',
    chartType: 'scatter',
    dataSource: null, // per-run, fetched by interactive figure
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
    chartType: 'bar',
    staticData: 'modelSizes',
    dataSource: null,
    defaultOptions: {
      xField: 'model_type',
      yField: 'param_count_M',
      colorField: 'scale',
      xLabel: 'Model',
      yLabel: 'Parameters (M)',
    },
    layout: 'body',
    section: '06-knowledge-distillation',
  },
  {
    id: 'fig-cka-similarity',
    number: 10,
    caption: 'CKA similarity between teacher and student layer representations.',
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
