/* Declarative panel definitions — adding a panel = adding an entry */

export const PANELS = [
    // --- Overview ---
    {
        id: 'overview',
        title: 'Overview',
        description: 'At-a-glance summary of experiment results',
        chartType: 'summary',
        dataSource: ['leaderboard.json', 'model_sizes.json', 'kd_transfer.json'],
        joinFn: (leaderboard, sizes, kd) => ({ leaderboard, sizes, kd }),
    },

    // --- Core Panels ---
    {
        id: 'leaderboard',
        title: 'Leaderboard',
        description: 'Best validation metrics per model configuration',
        chartType: 'table',
        dataSource: 'leaderboard.json',
        controls: [
            {
                type: 'select', id: 'lb-metric', label: 'Metric',
                options: [], // populated from metric_catalog
                catalogSource: true,
                fallback: ['f1', 'accuracy', 'precision', 'recall', 'auc', 'mcc'],
                default: 'f1',
                mapTo: 'metric',
            },
        ],
    },
    {
        id: 'dataset-comparison',
        title: 'Dataset Comparison',
        description: 'Metric across datasets per model configuration',
        chartType: 'bar',
        dataSource: 'leaderboard.json',
        controls: [
            {
                type: 'select', id: 'dc-metric', label: 'Metric',
                options: [],
                catalogSource: true,
                fallback: ['f1', 'accuracy', 'auc'],
                default: 'f1',
                mapTo: 'metric',
            },
        ],
    },
    {
        id: 'kd-transfer',
        title: 'Knowledge Distillation Transfer',
        description: 'Teacher vs Student performance (points above diagonal = student exceeds teacher)',
        chartType: 'scatter',
        dataSource: 'kd_transfer.json',
        chartConfig: {
            xField: 'teacher_value',
            yField: 'student_value',
            colorField: 'dataset',
            xLabel: 'Teacher F1',
            yLabel: 'Student F1',
            xDomain: [0, 1],
            yDomain: [0, 1],
            diagonalLine: true,
            tooltipFn: d => `${d.dataset} / ${d.model_type}<br>Teacher: ${(d.teacher_value ?? 0).toFixed(4)}<br>Student: ${(d.student_value ?? 0).toFixed(4)}`,
        },
        preFilter: data => data.filter(d => d.metric_name === 'f1'),
    },
    {
        id: 'training-curves',
        title: 'Training Curves',
        description: 'Per-epoch metrics over training',
        chartType: 'line',
        dataSource: null,  // dynamic — loaded per run
        dynamicLoader: 'training_curves',
        controls: [
            {
                type: 'select', id: 'tc-metric', label: 'Metric',
                options: [
                    { value: 'val_loss', label: 'Val Loss' },
                    { value: 'train_loss', label: 'Train Loss' },
                    { value: 'val_loss,train_loss', label: 'Train + Val Loss' },
                    { value: 'val_acc', label: 'Val Accuracy' },
                    { value: 'train_acc', label: 'Train Accuracy' },
                    { value: 'val_acc,train_acc', label: 'Train + Val Acc' },
                ],
                default: 'val_loss',
                mapTo: 'metric',
            },
            {
                type: 'select', id: 'tc-run', label: 'Run',
                options: [],  // populated from runs
                runSource: true,
                filterStages: ['autoencoder', 'curriculum', 'fusion'],
                mapTo: '_run',  // special: triggers data reload
            },
        ],
    },
    {
        id: 'training-compare',
        title: 'Training Curve Comparison',
        description: 'Overlay training curves from two runs to compare convergence',
        chartType: 'line',
        dataSource: null,
        dynamicLoader: 'training_curves_compare',
        renderMethod: 'renderMultiRun',
        controls: [
            {
                type: 'select', id: 'tcc-metric', label: 'Metric',
                options: [
                    { value: 'val_loss', label: 'Val Loss' },
                    { value: 'train_loss', label: 'Train Loss' },
                    { value: 'val_loss,train_loss', label: 'Train + Val Loss' },
                    { value: 'val_acc', label: 'Val Accuracy' },
                    { value: 'val_acc,train_acc', label: 'Train + Val Acc' },
                ],
                default: 'val_loss',
                mapTo: 'metric',
            },
            {
                type: 'select', id: 'tcc-runA', label: 'Run A',
                options: [],
                runSource: true,
                filterStages: ['autoencoder', 'curriculum', 'fusion'],
                mapTo: '_runA',
            },
            {
                type: 'select', id: 'tcc-runB', label: 'Run B',
                options: [],
                runSource: true,
                filterStages: ['autoencoder', 'curriculum', 'fusion'],
                mapTo: '_runB',
            },
        ],
    },
    {
        id: 'training-duration',
        title: 'Training Duration',
        description: 'Wall-clock training time per model configuration',
        chartType: 'bar',
        dataSource: 'runs.json',
        preTransform: (runs) => {
            return runs
                .filter(r => r.started_at && r.completed_at && r.status === 'complete')
                .map(r => {
                    const ms = new Date(r.completed_at) - new Date(r.started_at);
                    const minutes = Math.round(ms / 60000 * 10) / 10;
                    return {
                        dataset: r.dataset,
                        model_type: r.model_type,
                        scale: r.scale,
                        metric_name: 'duration_min',
                        best_value: minutes,
                        _stage: r.stage,
                    };
                });
        },
        controls: [
            {
                type: 'select', id: 'td-metric', label: 'Metric',
                options: [{ value: 'duration_min', label: 'Duration (min)' }],
                default: 'duration_min',
                mapTo: 'metric',
            },
            {
                type: 'select', id: 'td-stage', label: 'Stage',
                options: [
                    { value: '', label: 'All' },
                    { value: 'autoencoder', label: 'Autoencoder' },
                    { value: 'curriculum', label: 'Curriculum' },
                    { value: 'fusion', label: 'Fusion' },
                    { value: 'evaluation', label: 'Evaluation' },
                ],
                default: '',
                mapTo: '_stage_filter',
            },
        ],
    },
    {
        id: 'run-timeline',
        title: 'Run Timeline',
        description: 'Experiment runs over time',
        chartType: 'timeline',
        dataSource: 'runs.json',
    },

    // --- Model Analysis ---
    {
        id: 'graph-structure',
        title: 'CAN Bus Graph Structure',
        description: 'Force-directed visualization of CAN bus message graphs',
        chartType: 'force',
        dataSource: 'graph_samples.json',
        controls: [
            {
                type: 'select', id: 'gs-dataset', label: 'Dataset',
                options: [],  // populated from data
                datasetSource: true,
                mapTo: 'dataset',
            },
            {
                type: 'select', id: 'gs-label', label: 'Class',
                options: [
                    { value: '', label: 'All' },
                    { value: '0', label: 'Normal' },
                    { value: '1', label: 'Attack' },
                ],
                default: '',
                mapTo: 'label',
            },
            {
                type: 'select', id: 'gs-color', label: 'Color By',
                options: [
                    { value: 'can_id', label: 'CAN ID' },
                    { value: 'degree', label: 'Degree' },
                ],
                default: 'can_id',
                mapTo: 'colorBy',
            },
        ],
    },
    {
        id: 'model-comparison',
        title: 'Multi-Metric Model Comparison',
        description: 'Bubble chart: x/y = metrics, size = parameter count, color = model type',
        chartType: 'bubble',
        dataSource: ['leaderboard.json', 'model_sizes.json'],
        joinFn: (leaderboard, sizes) => {
            const sizeMap = new Map();
            sizes.forEach(s => sizeMap.set(`${s.model_type}_${s.scale}`, s));
            const f1Data = leaderboard.filter(d => d.metric_name === 'f1');
            const accData = leaderboard.filter(d => d.metric_name === 'accuracy');
            return f1Data.map(f1 => {
                const acc = accData.find(a =>
                    a.dataset === f1.dataset && a.model_type === f1.model_type && a.scale === f1.scale
                );
                const sz = sizeMap.get(`${f1.model_type}_${f1.scale}`) || {};
                return {
                    dataset: f1.dataset,
                    model_type: f1.model_type,
                    scale: f1.scale,
                    f1: f1.best_value,
                    accuracy: acc?.best_value ?? 0,
                    param_count: sz.param_count || 0,
                    param_count_M: sz.param_count_M || 0,
                };
            });
        },
        controls: [
            {
                type: 'select', id: 'mc-dataset', label: 'Dataset',
                options: [],
                datasetSource: true,
                mapTo: '_dataset_filter',
            },
        ],
    },
    {
        id: 'pareto-frontier',
        title: 'Pareto Frontier',
        description: 'F1 vs parameter count trade-off — dashed line shows Pareto-optimal configs',
        chartType: 'scatter',
        dataSource: ['leaderboard.json', 'model_sizes.json'],
        joinFn: (leaderboard, sizes) => {
            const sizeMap = new Map();
            sizes.forEach(s => sizeMap.set(`${s.model_type}_${s.scale}`, s));
            return leaderboard
                .filter(d => d.metric_name === 'f1')
                .map(d => {
                    const sz = sizeMap.get(`${d.model_type}_${d.scale}`) || {};
                    return {
                        dataset: d.dataset,
                        config: `${d.model_type}_${d.scale}${d.has_kd ? '_kd' : ''}`,
                        param_count_M: sz.param_count_M || 0,
                        f1: d.best_value ?? 0,
                    };
                })
                .filter(d => d.param_count_M > 0);
        },
        chartConfig: {
            xField: 'param_count_M',
            yField: 'f1',
            colorField: 'config',
            xLabel: 'Parameters (M)',
            yLabel: 'F1 Score',
            paretoLine: true,
            tooltipFn: d => `${d.config}<br>Dataset: ${d.dataset}<br>Params: ${d.param_count_M}M<br>F1: ${(d.f1 ?? 0).toFixed(4)}`,
        },
        controls: [
            {
                type: 'select', id: 'pf-dataset', label: 'Dataset',
                options: [],
                datasetSource: true,
                mapTo: '_dataset_filter',
            },
        ],
    },
    {
        id: 'model-predictions',
        title: 'Model Predictions Breakdown',
        description: 'Per-scenario metrics across model types',
        chartType: 'bar',
        dataSource: null,
        dynamicLoader: 'predictions',
        controls: [
            {
                type: 'select', id: 'mp-run', label: 'Run',
                options: [],
                evalRunSource: true,
                mapTo: '_run',
            },
            {
                type: 'select', id: 'mp-metric', label: 'Metric',
                options: [
                    { value: 'f1', label: 'F1' },
                    { value: 'accuracy', label: 'Accuracy' },
                    { value: 'auc', label: 'AUC' },
                    { value: 'precision', label: 'Precision' },
                    { value: 'recall', label: 'Recall' },
                ],
                default: 'f1',
                mapTo: 'metric',
            },
        ],
    },

    // --- Advanced Analysis ---
    {
        id: 'confusion-matrix',
        title: 'Confusion Matrix',
        description: 'Classification confusion matrix with counts and percentages',
        chartType: 'heatmap',
        dataSource: null,
        dynamicLoader: 'confusion_matrix',
        controls: [
            {
                type: 'select', id: 'cm-run', label: 'Run',
                options: [],
                evalRunSource: true,
                mapTo: '_run',
            },
            {
                type: 'select', id: 'cm-model', label: 'Model',
                options: [
                    { value: 'gat', label: 'GAT' },
                    { value: 'vgae', label: 'VGAE' },
                    { value: 'fusion', label: 'Fusion' },
                ],
                default: 'gat',
                mapTo: '_model',
            },
            {
                type: 'select', id: 'cm-scenario', label: 'Scenario',
                options: [
                    { value: 'val', label: 'Validation' },
                ],
                default: 'val',
                mapTo: '_scenario',
            },
        ],
    },
    {
        id: 'training-carpet',
        title: 'Training Curve Heatmap',
        description: 'Convergence patterns across all runs (rows=runs, cols=epochs, value=metric)',
        chartType: 'heatmap',
        dataSource: null,
        dynamicLoader: 'training_carpet',
        controls: [
            {
                type: 'select', id: 'tch-metric', label: 'Metric',
                options: [
                    { value: 'val_loss', label: 'Val Loss' },
                    { value: 'train_loss', label: 'Train Loss' },
                    { value: 'val_acc', label: 'Val Accuracy' },
                ],
                default: 'val_loss',
                mapTo: '_metric',
            },
        ],
    },
];
