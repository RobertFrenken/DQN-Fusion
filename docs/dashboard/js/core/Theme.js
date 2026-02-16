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

export const MARGIN = { top: 20, right: 30, bottom: 50, left: 60 };

export function colorScale(domain) {
    return d3.scaleOrdinal().domain(domain).range(COLORS);
}
