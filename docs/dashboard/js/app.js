/* KD-GAT Dashboard — Slim entry point using PanelManager */

// Import all chart types (side-effect: registers with Registry)
import './charts/TableChart.js';
import './charts/BarChart.js';
import './charts/ScatterChart.js';
import './charts/LineChart.js';
import './charts/TimelineChart.js';
import './charts/BubbleChart.js';
import './charts/ForceGraph.js';
import './charts/HistogramChart.js';
import './charts/SummaryChart.js';
import './charts/HeatmapChart.js';
import './charts/CurveChart.js';

// Import and initialize PanelManager
import { PanelManager } from './panels/PanelManager.js';

const manager = new PanelManager();
manager.init().catch(err => {
    console.error('Dashboard initialization failed:', err);
    document.querySelector('main').innerHTML =
        '<div class="no-data">Dashboard failed to load. Check console for details.</div>';
});
