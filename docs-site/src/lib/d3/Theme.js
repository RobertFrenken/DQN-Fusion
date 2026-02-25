/* Shared palette and scale helpers */

import * as d3 from 'd3';

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
    vgae: '#58a6ff',  // blue
    gat: '#3fb950',   // green
    dqn: '#d29922',   // amber
    fusion: '#bc8cff', // purple
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
    val_loss: '#f85149',    // red
    train_loss: '#d29922',  // amber
    val_acc: '#3fb950',     // green
    train_acc: '#58a6ff',   // blue
};

export const MARGIN = { top: 20, right: 30, bottom: 50, left: 60 };

/**
 * Build a D3 ordinal scale using a semantic color map, falling back to COLORS.
 * @param {string[]} domain - ordinal domain values
 * @param {Object} map - key→color mapping (e.g. MODEL_COLORS)
 */
export function semanticColorScale(domain, map) {
    const range = domain.map((d, i) => map[d] || COLORS[i % COLORS.length]);
    return d3.scaleOrdinal().domain(domain).range(range);
}

export function colorScale(domain) {
    return d3.scaleOrdinal().domain(domain).range(COLORS);
}
