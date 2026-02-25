/**
 * KD-GAT color palette and scale helpers.
 * Extracted from docs-site/src/lib/d3/Theme.js — standalone, no d3 dependency.
 */

export const COLORS = [
  '#58a6ff', '#3fb950', '#d29922', '#f85149',
  '#bc8cff', '#f778ba', '#79c0ff', '#56d364',
  '#ff9a5c', '#66d9ef', '#a9dc76', '#ffd866',
  '#ff6188', '#ab9df2', '#78dce8', '#e06c75',
  '#98c379', '#61afef', '#c678dd', '#d19a66',
];

export const STATUS_COLORS = {
  complete: '#3fb950',
  failed: '#f85149',
  running: '#d29922',
  unknown: '#8b949e',
};

export const LABEL_COLORS = {
  normal: '#3fb950',
  attack: '#f85149',
};

export const MODEL_COLORS = {
  vgae: '#58a6ff',
  gat: '#3fb950',
  dqn: '#d29922',
  fusion: '#bc8cff',
};

export const CONFIG_COLORS = {
  vgae_large: '#58a6ff',
  vgae_small: '#79c0ff',
  vgae_small_kd: '#a5d6ff',
  gat_large: '#3fb950',
  gat_small: '#56d364',
  gat_small_kd: '#7ee787',
  dqn_large: '#d29922',
  dqn_small: '#e3b341',
  dqn_small_kd: '#f0d070',
};

export const DATASET_COLORS = {
  hcrl_ch: '#58a6ff',
  hcrl_sa: '#3fb950',
  set_01: '#d29922',
  set_02: '#f85149',
  set_03: '#bc8cff',
  set_04: '#f778ba',
};

export const METRIC_COLORS = {
  val_loss: '#f85149',
  train_loss: '#d29922',
  val_acc: '#3fb950',
  train_acc: '#58a6ff',
};

export const MARGIN = { top: 20, right: 30, bottom: 50, left: 60 };

/**
 * Build a color range from a semantic map, falling back to COLORS.
 * @param {string[]} domain - ordinal domain values
 * @param {Object} map - key→color mapping (e.g., MODEL_COLORS)
 * @returns {string[]} color range array
 */
export function semanticColorRange(domain, map) {
  return domain.map((d, i) => map[d] || COLORS[i % COLORS.length]);
}
