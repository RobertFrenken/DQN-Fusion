/* KD-GAT Dashboard — D3.js chart functions */

const Charts = (() => {
    const margin = { top: 20, right: 30, bottom: 50, left: 60 };
    const colors = [
        '#58a6ff', '#3fb950', '#d29922', '#f85149',
        '#bc8cff', '#f778ba', '#79c0ff', '#56d364',
    ];

    function noData(container) {
        d3.select(container).html('<div class="no-data">No data available</div>');
    }

    /** Sortable leaderboard table */
    function leaderboard(container, data, metric) {
        const el = d3.select(container);
        el.html('');

        if (!data || data.length === 0) return noData(container);

        // Pivot: group by dataset+model+scale+has_kd, show selected metric
        const filtered = data.filter(d => d.metric_name === metric);
        if (filtered.length === 0) return noData(container);

        // Build rows keyed by config
        const rows = filtered.map(d => ({
            dataset: d.dataset,
            model: d.model,
            model_type: d.model_type,
            scale: d.scale,
            kd: d.has_kd ? 'Yes' : 'No',
            value: d.best_value,
        }));

        rows.sort((a, b) => b.value - a.value);

        const table = el.append('table');
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
                        return asc
                            ? d3.ascending(a[key], b[key])
                            : d3.descending(a[key], b[key]);
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
                    .classed('good', r.value >= 0.9)
                    .classed('ok', r.value >= 0.7 && r.value < 0.9)
                    .text(r.value != null ? r.value.toFixed(4) : '—');
            });
        }
        renderRows();
    }

    /** Grouped bar chart: metric across datasets per model config */
    function datasetComparison(container, data, metric) {
        const el = d3.select(container);
        el.html('');

        const filtered = data.filter(d => d.metric_name === metric);
        if (filtered.length === 0) return noData(container);

        const width = el.node().clientWidth - margin.left - margin.right;
        const height = 350 - margin.top - margin.bottom;

        const svg = el.append('svg')
            .attr('width', width + margin.left + margin.right)
            .attr('height', height + margin.top + margin.bottom)
            .append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);

        const datasets = [...new Set(filtered.map(d => d.dataset))];
        const configs = [...new Set(filtered.map(d => `${d.model_type}_${d.scale}`))];

        const x0 = d3.scaleBand().domain(datasets).range([0, width]).padding(0.2);
        const x1 = d3.scaleBand().domain(configs).range([0, x0.bandwidth()]).padding(0.05);
        const y = d3.scaleLinear().domain([0, 1]).range([height, 0]);
        const color = d3.scaleOrdinal().domain(configs).range(colors);

        // Axes
        svg.append('g')
            .attr('class', 'axis')
            .attr('transform', `translate(0,${height})`)
            .call(d3.axisBottom(x0));

        svg.append('g')
            .attr('class', 'axis')
            .call(d3.axisLeft(y).ticks(5));

        // Grid
        svg.append('g')
            .attr('class', 'grid')
            .call(d3.axisLeft(y).ticks(5).tickSize(-width).tickFormat(''));

        // Tooltip
        const tooltip = d3.select('body').append('div').attr('class', 'tooltip');

        // Bars
        datasets.forEach(ds => {
            const dsData = filtered.filter(d => d.dataset === ds);
            configs.forEach(cfg => {
                const d = dsData.find(d => `${d.model_type}_${d.scale}` === cfg);
                if (!d) return;
                svg.append('rect')
                    .attr('x', x0(ds) + x1(cfg))
                    .attr('y', y(d.best_value))
                    .attr('width', x1.bandwidth())
                    .attr('height', height - y(d.best_value))
                    .attr('fill', color(cfg))
                    .attr('rx', 2)
                    .on('mouseover', (event) => {
                        tooltip.style('opacity', 1)
                            .html(`${cfg}<br>${ds}: ${d.best_value.toFixed(4)}`);
                    })
                    .on('mousemove', (event) => {
                        tooltip.style('left', (event.pageX + 10) + 'px')
                            .style('top', (event.pageY - 20) + 'px');
                    })
                    .on('mouseout', () => tooltip.style('opacity', 0));
            });
        });

        // Legend
        const legend = svg.append('g')
            .attr('transform', `translate(${width - 120}, 0)`);
        configs.forEach((cfg, i) => {
            const g = legend.append('g').attr('transform', `translate(0, ${i * 18})`);
            g.append('rect').attr('width', 12).attr('height', 12).attr('fill', color(cfg)).attr('rx', 2);
            g.append('text').attr('x', 16).attr('y', 10).attr('fill', '#c9d1d9').style('font-size', '11px').text(cfg);
        });
    }

    /** Scatter plot: teacher F1 vs student F1 */
    function kdTransfer(container, data) {
        const el = d3.select(container);
        el.html('');

        if (!data || data.length === 0) return noData(container);

        const f1Data = data.filter(d => d.metric_name === 'f1');
        if (f1Data.length === 0) return noData(container);

        const width = el.node().clientWidth - margin.left - margin.right;
        const height = 350 - margin.top - margin.bottom;

        const svg = el.append('svg')
            .attr('width', width + margin.left + margin.right)
            .attr('height', height + margin.top + margin.bottom)
            .append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);

        const x = d3.scaleLinear().domain([0, 1]).range([0, width]);
        const y = d3.scaleLinear().domain([0, 1]).range([height, 0]);

        svg.append('g').attr('class', 'axis').attr('transform', `translate(0,${height})`).call(d3.axisBottom(x));
        svg.append('g').attr('class', 'axis').call(d3.axisLeft(y));

        // 45-degree reference line
        svg.append('line')
            .attr('x1', x(0)).attr('y1', y(0))
            .attr('x2', x(1)).attr('y2', y(1))
            .attr('stroke', '#30363d').attr('stroke-dasharray', '4,4');

        // Axis labels
        svg.append('text').attr('x', width / 2).attr('y', height + 40)
            .attr('text-anchor', 'middle').attr('fill', '#8b949e').style('font-size', '12px')
            .text('Teacher F1');
        svg.append('text').attr('transform', 'rotate(-90)').attr('x', -height / 2).attr('y', -45)
            .attr('text-anchor', 'middle').attr('fill', '#8b949e').style('font-size', '12px')
            .text('Student F1');

        const tooltip = d3.select('body').append('div').attr('class', 'tooltip');
        const dsColor = d3.scaleOrdinal().range(colors);

        svg.selectAll('circle')
            .data(f1Data)
            .join('circle')
            .attr('cx', d => x(d.teacher_value || 0))
            .attr('cy', d => y(d.student_value || 0))
            .attr('r', 6)
            .attr('fill', d => dsColor(d.dataset))
            .attr('opacity', 0.8)
            .attr('stroke', '#fff')
            .attr('stroke-width', 1)
            .on('mouseover', (event, d) => {
                tooltip.style('opacity', 1)
                    .html(`${d.dataset} / ${d.model_type}<br>Teacher: ${(d.teacher_value||0).toFixed(4)}<br>Student: ${(d.student_value||0).toFixed(4)}`);
            })
            .on('mousemove', (event) => {
                tooltip.style('left', (event.pageX + 10) + 'px')
                    .style('top', (event.pageY - 20) + 'px');
            })
            .on('mouseout', () => tooltip.style('opacity', 0));
    }

    /** Line chart: training curves */
    function trainingCurves(container, data, metric) {
        const el = d3.select(container);
        el.html('');

        if (!data || data.length === 0) return noData(container);

        const filtered = data.filter(d => d.metric_name === metric);
        if (filtered.length === 0) return noData(container);

        const width = el.node().clientWidth - margin.left - margin.right;
        const height = 300 - margin.top - margin.bottom;

        const svg = el.append('svg')
            .attr('width', width + margin.left + margin.right)
            .attr('height', height + margin.top + margin.bottom)
            .append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);

        const x = d3.scaleLinear()
            .domain(d3.extent(filtered, d => d.epoch))
            .range([0, width]);
        const y = d3.scaleLinear()
            .domain(d3.extent(filtered, d => d.value))
            .nice()
            .range([height, 0]);

        svg.append('g').attr('class', 'axis').attr('transform', `translate(0,${height})`).call(d3.axisBottom(x).ticks(10));
        svg.append('g').attr('class', 'axis').call(d3.axisLeft(y));
        svg.append('g').attr('class', 'grid').call(d3.axisLeft(y).ticks(5).tickSize(-width).tickFormat(''));

        const line = d3.line()
            .x(d => x(d.epoch))
            .y(d => y(d.value));

        svg.append('path')
            .datum(filtered)
            .attr('fill', 'none')
            .attr('stroke', colors[0])
            .attr('stroke-width', 2)
            .attr('d', line);

        // Axis labels
        svg.append('text').attr('x', width / 2).attr('y', height + 40)
            .attr('text-anchor', 'middle').attr('fill', '#8b949e').style('font-size', '12px')
            .text('Epoch');
        svg.append('text').attr('transform', 'rotate(-90)').attr('x', -height / 2).attr('y', -45)
            .attr('text-anchor', 'middle').attr('fill', '#8b949e').style('font-size', '12px')
            .text(metric);
    }

    /** Run timeline: x=date, y=dataset, color=status */
    function runTimeline(container, data) {
        const el = d3.select(container);
        el.html('');

        const withDates = data.filter(d => d.started_at);
        if (withDates.length === 0) return noData(container);

        const width = el.node().clientWidth - margin.left - margin.right;
        const datasets = [...new Set(withDates.map(d => d.dataset))];
        const height = Math.max(200, datasets.length * 30) - margin.top - margin.bottom;

        const svg = el.append('svg')
            .attr('width', width + margin.left + margin.right)
            .attr('height', height + margin.top + margin.bottom)
            .append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);

        const parseTime = d3.timeParse('%Y-%m-%dT%H:%M:%S%Z');
        const parseTimeAlt = d3.timeParse('%Y-%m-%dT%H:%M:%S');
        withDates.forEach(d => {
            d._date = parseTime(d.started_at) || parseTimeAlt(d.started_at) || new Date(d.started_at);
        });

        const x = d3.scaleTime()
            .domain(d3.extent(withDates, d => d._date))
            .range([0, width]);
        const y = d3.scaleBand()
            .domain(datasets)
            .range([0, height])
            .padding(0.3);

        const statusColor = { complete: '#3fb950', failed: '#f85149', running: '#d29922', unknown: '#8b949e' };

        svg.append('g').attr('class', 'axis').attr('transform', `translate(0,${height})`).call(d3.axisBottom(x).ticks(6));
        svg.append('g').attr('class', 'axis').call(d3.axisLeft(y));

        const tooltip = d3.select('body').append('div').attr('class', 'tooltip');

        svg.selectAll('circle')
            .data(withDates)
            .join('circle')
            .attr('cx', d => x(d._date))
            .attr('cy', d => y(d.dataset) + y.bandwidth() / 2)
            .attr('r', 5)
            .attr('fill', d => statusColor[d.status] || statusColor.unknown)
            .attr('opacity', 0.8)
            .on('mouseover', (event, d) => {
                tooltip.style('opacity', 1)
                    .html(`${d.run_id}<br>${d.stage} / ${d.status}<br>${d.started_at}`);
            })
            .on('mousemove', (event) => {
                tooltip.style('left', (event.pageX + 10) + 'px')
                    .style('top', (event.pageY - 20) + 'px');
            })
            .on('mouseout', () => tooltip.style('opacity', 0));
    }

    return { leaderboard, datasetComparison, kdTransfer, trainingCurves, runTimeline };
})();
