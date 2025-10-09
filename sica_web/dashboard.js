/**
 * SICA-Style Trading Dashboard JavaScript
 * Handles command execution, callgraph visualization, and event streaming
 */

const API_BASE = 'http://localhost:8082';
let ws = null;
let reconnectInterval = null;

// Initialize dashboard
document.addEventListener('DOMContentLoaded', () => {
    initWebSocket();
    loadStats();
    setInterval(loadStats, 5000); // Update stats every 5 seconds
    setInterval(pollCallgraph, 2000); // Poll callgraph every 2 seconds
});

// WebSocket connection for real-time events
function initWebSocket() {
    ws = new WebSocket('ws://localhost:8082/ws/events');

    ws.onopen = () => {
        console.log('WebSocket connected');
        addEventLog('system', 'Connected to event stream');
        if (reconnectInterval) {
            clearInterval(reconnectInterval);
            reconnectInterval = null;
        }
    };

    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            handleEvent(data);
        } catch (e) {
            console.error('Error parsing event:', e);
        }
    };

    ws.onclose = () => {
        console.log('WebSocket disconnected');
        addEventLog('system', 'Disconnected from event stream');
        // Attempt reconnect
        if (!reconnectInterval) {
            reconnectInterval = setInterval(() => {
                console.log('Attempting to reconnect...');
                initWebSocket();
            }, 3000);
        }
    };

    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
    };
}

// Handle incoming events
function handleEvent(event) {
    const type = event.type || 'info';
    const content = event.content || JSON.stringify(event);
    addEventLog(type, content);
}

// Add event to log
function addEventLog(type, message) {
    const logContainer = document.getElementById('event-log');
    const now = new Date();
    const timestamp = now.toLocaleTimeString();

    const eventItem = document.createElement('div');
    eventItem.className = 'event-item';

    let typeColor = 'text-gray-600';
    if (type === 'error') typeColor = 'text-red-600';
    if (type === 'success') typeColor = 'text-green-600';
    if (type === 'system') typeColor = 'text-blue-600';

    eventItem.innerHTML = `
        <span class="event-time">[${timestamp}]</span>
        <span class="${typeColor}">${message}</span>
    `;

    // Clear "waiting" message if it exists
    if (logContainer.querySelector('.text-gray-500')) {
        logContainer.innerHTML = '';
    }

    logContainer.appendChild(eventItem);
    logContainer.scrollTop = logContainer.scrollHeight;

    // Limit to 100 events
    while (logContainer.children.length > 100) {
        logContainer.removeChild(logContainer.firstChild);
    }
}

// Clear event log
function clearEventLog() {
    const logContainer = document.getElementById('event-log');
    logContainer.innerHTML = '<div class="event-item text-gray-500"><span class="event-time">[--:--:--]</span>Waiting for events...</div>';
}

// Load system statistics
async function loadStats() {
    try {
        const response = await fetch(`${API_BASE}/api/stats`);
        const data = await response.json();

        document.getElementById('total-cells').textContent = data.total_cells || '--';
        document.getElementById('best-fitness').textContent = data.best_fitness || '--';
        document.getElementById('avg-fitness').textContent = data.avg_fitness || '--';
    } catch (error) {
        console.error('Error loading stats:', error);
    }
}

// Poll callgraph for updates
async function pollCallgraph() {
    try {
        const response = await fetch(`${API_BASE}/api/callgraph`);
        const data = await response.json();

        if (data.nodes && Object.keys(data.nodes).length > 0 && data.root_id) {
            renderCallgraph(data.nodes, data.root_id);
        }
    } catch (error) {
        // Silently fail - no callgraph available
    }
}

// Render callgraph/execution tree
function renderCallgraph(nodesDict, rootId) {
    const treeContainer = document.getElementById('execution-tree');
    treeContainer.innerHTML = '';

    if (rootId && nodesDict[rootId]) {
        renderNode(nodesDict[rootId], nodesDict, treeContainer, 0);
    }
}

function renderNode(node, nodesDict, container, depth) {
    const nodeEl = document.createElement('div');

    // Determine status class
    let statusClass = '';
    if (node.success === true) {
        statusClass = 'success';
    } else if (node.success === false) {
        statusClass = 'error';
    } else if (node.started_at && !node.completed_at) {
        statusClass = 'running';
    }

    nodeEl.className = `tree-node ${statusClass}`;
    nodeEl.style.marginLeft = `${depth * 20}px`;

    // Choose icon based on status
    const icon = node.success === true ? '✓' :
                 node.success === false ? '✗' :
                 node.completed_at ? '●' : '▶';

    const statusText = (!node.completed_at && node.started_at) ? '(running)' : '';

    // Format duration if available
    let durationText = '';
    if (node.duration_seconds) {
        durationText = ` (${node.duration_seconds.toFixed(2)}s)`;
    }

    // Format cost if available
    let costText = '';
    if (node.cost) {
        costText = ` | $${node.cost.toFixed(4)}`;
    }

    nodeEl.innerHTML = `
        <div class="font-semibold">${icon} ${node.name} ${statusText}</div>
        <div class="text-xs text-gray-600 mt-1">
            ${durationText}${costText}
            ${node.token_count ? ` | ${node.token_count} tokens` : ''}
        </div>
        ${node.events && node.events.length > 0 ? `<div class="text-xs text-blue-600 mt-1">${node.events.length} events</div>` : ''}
    `;

    container.appendChild(nodeEl);

    // Render children recursively
    if (node.children && node.children.length > 0) {
        node.children.forEach(childId => {
            if (nodesDict[childId]) {
                renderNode(nodesDict[childId], nodesDict, container, depth + 1);
            }
        });
    }
}

// Update status message
function updateStatus(message, type = 'info') {
    const statusEl = document.getElementById('action-status');
    statusEl.textContent = message;

    if (type === 'error') {
        statusEl.className = 'mt-3 text-sm text-red-600';
    } else if (type === 'success') {
        statusEl.className = 'mt-3 text-sm text-green-600';
    } else if (type === 'info') {
        statusEl.className = 'mt-3 text-sm text-blue-600';
    } else {
        statusEl.className = 'mt-3 text-sm text-gray-600';
    }
}

// Command: Trading-Evolve
async function runTradingEvolve() {
    updateStatus('Starting trading-evolve (offline evolution with 1000 generations)...', 'info');
    addEventLog('command', 'Trading-Evolve started');

    try {
        const response = await fetch(`${API_BASE}/api/command/trading-evolve`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                generations: 1000,
                fitness_goal: 50.0
            })
        });

        const data = await response.json();

        if (response.ok) {
            updateStatus(data.message, 'success');
            addEventLog('success', data.message);
        } else {
            updateStatus('Error: ' + (data.detail || 'Failed to start trading-evolve'), 'error');
            addEventLog('error', data.detail || 'Failed to start trading-evolve');
        }
    } catch (error) {
        updateStatus('Error: ' + error.message, 'error');
        addEventLog('error', error.message);
    }
}

// Command: Trading-Learn
async function runTradingLearn() {
    updateStatus('Starting trading-learn (LLM-guided evolution)...', 'info');
    addEventLog('command', 'Trading-Learn started');

    try {
        const response = await fetch(`${API_BASE}/api/command/trading-learn`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                num_strategies: 5,
                confidence: 1.0,
                use_local: true  // Use Ollama by default
            })
        });

        const data = await response.json();

        if (response.ok) {
            updateStatus(data.message, 'success');
            addEventLog('success', data.message);
        } else {
            updateStatus('Error: ' + (data.detail || 'Failed to start trading-learn'), 'error');
            addEventLog('error', data.detail || 'Failed to start trading-learn');
        }
    } catch (error) {
        updateStatus('Error: ' + error.message, 'error');
        addEventLog('error', error.message);
    }
}

// Command: Trading-Test
async function runTradingTest() {
    updateStatus('Starting trading-test (backtesting best strategy)...', 'info');
    addEventLog('command', 'Trading-Test started');

    try {
        const response = await fetch(`${API_BASE}/api/command/trading-test`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                cell_id: null  // Test best cell
            })
        });

        const data = await response.json();

        if (response.ok) {
            updateStatus(data.message, 'success');
            addEventLog('success', data.message);
        } else {
            updateStatus('Error: ' + (data.detail || 'Failed to start trading-test'), 'error');
            addEventLog('error', data.detail || 'Failed to start trading-test');
        }
    } catch (error) {
        updateStatus('Error: ' + error.message, 'error');
        addEventLog('error', error.message);
    }
}

// Show query modal
function showQueryModal() {
    document.getElementById('query-modal').classList.remove('hidden');
}

// Close query modal
function closeQueryModal() {
    document.getElementById('query-modal').classList.add('hidden');
}

// Execute cell query
async function executeCellQuery() {
    const limit = document.getElementById('query-limit').value;
    const minTrades = document.getElementById('query-min-trades').value;

    closeQueryModal();
    updateStatus(`Querying top ${limit} cells...`, 'info');

    try {
        const response = await fetch(`${API_BASE}/api/command/query-cells?limit=${limit}&min_trades=${minTrades}`);
        const data = await response.json();

        if (response.ok) {
            updateStatus(`Found ${data.count} cells`, 'success');
            addEventLog('success', `Query returned ${data.count} cells`);

            // Switch to cells tab and display results
            showTab('cells');
            renderCellTable(data.cells);
        } else {
            updateStatus('Error: ' + (data.detail || 'Query failed'), 'error');
            addEventLog('error', data.detail || 'Query failed');
        }
    } catch (error) {
        updateStatus('Error: ' + error.message, 'error');
        addEventLog('error', error.message);
    }
}

// Render cell table
function renderCellTable(cells) {
    const tbody = document.getElementById('cell-table-body');
    tbody.innerHTML = '';

    if (!cells || cells.length === 0) {
        tbody.innerHTML = '<tr><td colspan="6" class="px-4 py-8 text-center text-gray-500">No cells found</td></tr>';
        return;
    }

    cells.forEach(cell => {
        const row = document.createElement('tr');
        row.className = 'hover:bg-gray-50';

        const source = cell.llm_name ? '🤖 LLM' : '🧬 Evolution';
        const fitnessClass = cell.fitness > 0 ? 'text-green-600' : 'text-red-600';

        row.innerHTML = `
            <td class="px-4 py-3 text-sm">${cell.cell_id}</td>
            <td class="px-4 py-3 text-sm">${cell.generation}</td>
            <td class="px-4 py-3 text-sm font-semibold ${fitnessClass}">${cell.fitness.toFixed(2)}</td>
            <td class="px-4 py-3 text-sm">${cell.total_trades}</td>
            <td class="px-4 py-3 text-sm mono text-xs text-gray-600">${cell.dsl_genome}</td>
            <td class="px-4 py-3 text-sm">${source}</td>
        `;

        tbody.appendChild(row);
    });
}

// Tab switching
function showTab(tab) {
    // Update tab buttons
    document.getElementById('tab-execution').className = tab === 'execution'
        ? 'tab-button border-b-2 border-blue-500 py-4 px-1 text-sm font-medium text-blue-600'
        : 'tab-button border-b-2 border-transparent py-4 px-1 text-sm font-medium text-gray-500 hover:border-gray-300 hover:text-gray-700';

    document.getElementById('tab-cells').className = tab === 'cells'
        ? 'tab-button border-b-2 border-blue-500 py-4 px-1 text-sm font-medium text-blue-600'
        : 'tab-button border-b-2 border-transparent py-4 px-1 text-sm font-medium text-gray-500 hover:border-gray-300 hover:text-gray-700';

    // Update views
    document.getElementById('view-execution').style.display = tab === 'execution' ? 'block' : 'none';
    document.getElementById('view-cells').style.display = tab === 'cells' ? 'block' : 'none';

    // Load cells if switching to cells tab
    if (tab === 'cells') {
        executeCellQuery();
    }
}
