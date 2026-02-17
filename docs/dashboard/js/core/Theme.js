/* Shared palette and scale helpers */

export const COLORS = [
    '#58a6ff', '#3fb950', '#d29922', '#f85149',
    '#bc8cff', '#f778ba', '#79c0ff', '#56d364',
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
