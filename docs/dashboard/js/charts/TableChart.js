/* Sortable leaderboard table */

import { BaseChart } from '../core/BaseChart.js';
import { register } from '../core/Registry.js';

export class TableChart extends BaseChart {
    _setupSVG() {
        // Tables don't use SVG — override to clear only
        this.container.html('');
        this._width = 0;
        this._height = 0;
    }

    render(data, options = {}) {
        const metric = options.metric || 'f1';
        const filtered = data.filter(d => d.metric_name === metric);
        if (filtered.length === 0) return this._showNoData();

        const rows = filtered.map(d => ({
            dataset: d.dataset,
            model: d.model,
            model_type: d.model_type,
            scale: d.scale,
            kd: d.has_kd ? 'Yes' : 'No',
            value: d.best_value ?? null,
        }));

        rows.sort((a, b) => (b.value ?? -Infinity) - (a.value ?? -Infinity));

        const table = this.container.append('table');
        const thead = table.append('thead').append('tr');
        const cols = ['dataset', 'model', 'model_type', 'scale', 'kd', metric];

        cols.forEach(c => {
            thead.append('th')
                .text(c)
                .classed('num', c === metric)
                .on('click', () => {
                    const asc = thead.attr('data-sort') === c;
                    thead.attr('data-sort', asc ? '' : c);
                    rows.sort((a, b) => {
                        const key = c === metric ? 'value' : c;
                        return asc ? d3.ascending(a[key], b[key]) : d3.descending(a[key], b[key]);
                    });
                    renderRows();
                });
        });

        const tbody = table.append('tbody');

        function renderRows() {
            tbody.selectAll('tr').remove();
            rows.forEach(r => {
                const tr = tbody.append('tr');
                tr.append('td').text(r.dataset);
                tr.append('td').text(r.model);
                tr.append('td').text(r.model_type);
                tr.append('td').text(r.scale);
                tr.append('td').text(r.kd);
                tr.append('td')
                    .classed('num', true)
                    .classed('good', r.value != null && r.value >= 0.9)
                    .classed('ok', r.value != null && r.value >= 0.7 && r.value < 0.9)
                    .text(r.value != null ? r.value.toFixed(4) : '\u2014');
            });
        }
        renderRows();
    }
}

register('table', TableChart);
