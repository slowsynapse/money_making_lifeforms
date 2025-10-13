// Trading Evolution Dashboard
const API_BASE = 'http://localhost:8081';

let allCells = [];
let selectedCell = null;
let fitnessChart = null;
let currentFilter = null; // null = all, true = LLM only, false = evolution only
let activityLog = [];
const MAX_LOG_ENTRIES = 100;
let selectedDish = null; // Currently selected dish name

// Initialize dashboard
async function init() {
    console.log('Initializing dashboard...');
    logActivity('system', 'Dashboard initialized');
    await loadDishes();
    await loadCells();
    setupEventListeners();
    setupActivityMonitoring();
}

// Load dishes into selector
async function loadDishes() {
    try {
        const response = await fetch(`${API_BASE}/dishes`);
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

        const data = await response.json();
        const selector = document.getElementById('dish-selector');

        // Clear existing options except "Most Recent"
        selector.innerHTML = '<option value="">Most Recent</option>';

        // Add dish options
        data.dishes.forEach(dish => {
            const option = document.createElement('option');
            option.value = dish.dish_name;
            option.textContent = `🧫 ${dish.dish_name} (${dish.total_cells} cells, gen ${dish.total_generations})`;
            selector.appendChild(option);
        });

        console.log(`Loaded ${data.dishes.length} dishes`);
    } catch (error) {
        console.error('Failed to load dishes:', error);
        logActivity('error', `Failed to load dishes: ${error.message}`);
    }
}

// Activity logging functions
function logActivity(type, message, data = null) {
    const timestamp = new Date().toLocaleTimeString();
    const entry = { type, message, timestamp, data };
    activityLog.unshift(entry);

    // Keep only last MAX_LOG_ENTRIES
    if (activityLog.length > MAX_LOG_ENTRIES) {
        activityLog.pop();
    }

    updateActivityLog();
}

function updateActivityLog() {
    const logEl = document.getElementById('activity-log');
    const colors = {
        'system': 'text-blue-400',
        'api': 'text-green-400',
        'llm': 'text-purple-400',
        'evolution': 'text-yellow-400',
        'error': 'text-red-400',
        'success': 'text-emerald-400'
    };

    if (activityLog.length === 0) {
        logEl.innerHTML = '<div class="text-gray-500">Waiting for activity...</div>';
        return;
    }

    logEl.innerHTML = activityLog.map(entry => {
        const color = colors[entry.type] || 'text-gray-400';
        return `<div class="${color} mb-1">
            <span class="text-gray-600">[${entry.timestamp}]</span>
            <span class="text-gray-500">${entry.type.toUpperCase()}:</span>
            ${escapeHtml(entry.message)}
        </div>`;
    }).join('');

    // Auto-scroll to top (newest entries)
    logEl.scrollTop = 0;
}

// Setup activity monitoring
function setupActivityMonitoring() {
    // Monitor API calls via fetch wrapper
    window.originalFetch = window.fetch;
    window.fetch = async function(...args) {
        const url = args[0];
        const options = args[1] || {};

        // Skip logging for output polling OR when API logging is disabled
        const skipLogging = (isPollingActive && url.includes('/learn/output')) || !apiLoggingEnabled;

        // Log API calls (except when skipped)
        if (url.includes(API_BASE) && !skipLogging) {
            const method = options.method || 'GET';
            const endpoint = url.replace(API_BASE, '');
            logActivity('api', `${method} ${endpoint}`);
        }

        try {
            const response = await window.originalFetch.apply(this, args);

            // Log response status (except output polling)
            if (url.includes(API_BASE) && !skipLogging) {
                const endpoint = url.replace(API_BASE, '');
                if (response.ok) {
                    logActivity('success', `✓ ${endpoint} (${response.status})`);
                } else {
                    logActivity('error', `✗ ${endpoint} (${response.status})`);
                }
            }

            return response;
        } catch (error) {
            if (url.includes(API_BASE) && !skipLogging) {
                logActivity('error', `✗ Request failed: ${error.message}`);
            }
            throw error;
        }
    };
}

// Load cells from API
async function loadCells(hasLlm = null) {
    try {
        console.log('Fetching cells from API...');
        let url = `${API_BASE}/cells/top/100`;
        const params = new URLSearchParams();

        if (hasLlm !== null) {
            params.append('has_llm', hasLlm);
        }
        if (selectedDish) {
            params.append('dish', selectedDish);
        }

        if (params.toString()) {
            url += `?${params.toString()}`;
        }

        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        allCells = data.cells;

        console.log(`Loaded ${allCells.length} cells ${selectedDish ? `from dish "${selectedDish}"` : '(most recent)'}`);

        // Update stats
        updateStats();

        // Render cells table
        renderCellsTable();

        // Render fitness chart
        renderFitnessChart();

    } catch (error) {
        console.error('Failed to load cells:', error);
        document.getElementById('cell-table-body').innerHTML = `
            <tr>
                <td colspan="6" class="px-4 py-8 text-center text-red-600">
                    Failed to load cells. Make sure the API server is running on port 8081.
                </td>
            </tr>
        `;
    }
}

// Update header stats
function updateStats() {
    const totalCells = allCells.length;
    const bestFitness = allCells.length > 0 ? Math.max(...allCells.map(c => c.fitness)) : 0;
    const avgFitness = allCells.length > 0
        ? allCells.reduce((sum, c) => sum + c.fitness, 0) / allCells.length
        : 0;

    document.getElementById('total-cells').textContent = totalCells;
    document.getElementById('best-fitness').textContent = `$${bestFitness.toFixed(2)}`;
    document.getElementById('avg-fitness').textContent = `$${avgFitness.toFixed(2)}`;
}

// Render cells table
function renderCellsTable() {
    const tbody = document.getElementById('cell-table-body');

    if (allCells.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="6" class="px-4 py-8 text-center text-gray-500">No cells found</td>
            </tr>
        `;
        return;
    }

    tbody.innerHTML = allCells.map(cell => `
        <tr class="cell-row" data-cell-id="${cell.cell_id}">
            <td class="px-4 py-3 text-sm font-medium text-gray-900">#${cell.cell_id}</td>
            <td class="px-4 py-3 text-sm text-gray-600">${cell.generation}</td>
            <td class="px-4 py-3 text-sm ${cell.fitness >= 0 ? 'fitness-positive' : 'fitness-negative'}">
                $${cell.fitness.toFixed(2)}
            </td>
            <td class="px-4 py-3 text-sm text-gray-600">${cell.total_trades}</td>
            <td class="px-4 py-3 text-sm text-gray-700">
                <div class="genome-text" title="${escapeHtml(cell.dsl_genome)}">
                    ${escapeHtml(cell.dsl_genome)}
                </div>
            </td>
            <td class="px-4 py-3">
                <span class="inline-flex px-2 py-1 text-xs font-medium rounded-full ${
                    cell.status === 'online' && cell.fitness >= 0
                        ? 'bg-green-100 text-green-800'
                        : 'bg-gray-100 text-gray-800'
                }">
                    ${cell.status === 'online' && cell.fitness >= 0 ? '\ud83d\udfe2 alive' : '\ud83d\udc80 dead'}
                </span>
            </td>
        </tr>
    `).join('');
}

// Render fitness chart
function renderFitnessChart() {
    const ctx = document.getElementById('fitness-chart').getContext('2d');

    // Group cells by generation
    const generations = [...new Set(allCells.map(c => c.generation))].sort((a, b) => a - b);
    const fitnessData = generations.map(gen => {
        const cellsInGen = allCells.filter(c => c.generation === gen);
        const maxFitness = Math.max(...cellsInGen.map(c => c.fitness));
        const avgFitness = cellsInGen.reduce((sum, c) => sum + c.fitness, 0) / cellsInGen.length;
        return { gen, maxFitness, avgFitness };
    });

    if (fitnessChart) {
        fitnessChart.destroy();
    }

    fitnessChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: fitnessData.map(d => `Gen ${d.gen}`),
            datasets: [
                {
                    label: 'Best Fitness',
                    data: fitnessData.map(d => d.maxFitness),
                    borderColor: 'rgb(34, 197, 94)',
                    backgroundColor: 'rgba(34, 197, 94, 0.1)',
                    borderWidth: 2,
                    tension: 0.1
                },
                {
                    label: 'Average Fitness',
                    data: fitnessData.map(d => d.avgFitness),
                    borderColor: 'rgb(59, 130, 246)',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    borderWidth: 2,
                    tension: 0.1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    callbacks: {
                        label: function(context) {
                            return `${context.dataset.label}: $${context.parsed.y.toFixed(2)}`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    ticks: {
                        callback: function(value) {
                            return '$' + value.toFixed(2);
                        }
                    }
                }
            }
        }
    });
}

// Load and display cell details
async function loadCellDetails(cellId) {
    try {
        console.log(`Loading details for cell ${cellId}...`);

        // Build URLs with dish parameter if selected
        const dishParam = selectedDish ? `?dish=${selectedDish}` : '';

        // Fetch cell details and lineage in parallel
        const [cellResponse, lineageResponse, phenotypesResponse] = await Promise.all([
            fetch(`${API_BASE}/cell/${cellId}${dishParam}`),
            fetch(`${API_BASE}/cell/${cellId}/lineage${dishParam}`),
            fetch(`${API_BASE}/cell/${cellId}/phenotypes${dishParam}`)
        ]);

        const cell = await cellResponse.json();
        const lineageData = await lineageResponse.json();
        const phenotypesData = await phenotypesResponse.json();

        renderCellDetails(cell, lineageData.lineage, phenotypesData.phenotypes);

        // Render the D3.js lineage visualization
        if (typeof renderLineageTree === 'function' && lineageData.lineage) {
            renderLineageTree(lineageData.lineage);
        }

    } catch (error) {
        console.error('Failed to load cell details:', error);
        document.getElementById('cell-details-content').innerHTML = `
            <p class="text-sm text-red-600 text-center py-8">Failed to load cell details</p>
        `;
    }
}

// Render cell details panel
function renderCellDetails(cell, lineage, phenotypes) {
    const content = document.getElementById('cell-details-content');

    content.innerHTML = `
        <!-- Cell Header -->
        <div class="mb-4">
            <div class="flex items-center justify-between mb-2">
                <h3 class="text-xl font-bold text-gray-900">Cell #${cell.cell_id}</h3>
                <span class="inline-flex px-2 py-1 text-xs font-medium rounded-full ${
                    cell.status === 'online' && cell.fitness >= 0
                        ? 'bg-green-100 text-green-800'
                        : 'bg-gray-100 text-gray-800'
                }">
                    ${cell.status === 'online' && cell.fitness >= 0 ? '\ud83d\udfe2 alive' : '\ud83d\udc80 dead'}
                </span>
            </div>
            <div class="grid grid-cols-2 gap-2 text-sm">
                <div>
                    <span class="text-gray-600">Generation:</span>
                    <span class="font-medium text-gray-900">${cell.generation}</span>
                </div>
                <div>
                    <span class="text-gray-600">Fitness:</span>
                    <span class="font-medium ${cell.fitness >= 0 ? 'text-green-600' : 'text-red-600'}">
                        $${cell.fitness.toFixed(2)}
                    </span>
                </div>
            </div>
        </div>

        <!-- Genome -->
        <div class="mb-4">
            <h4 class="text-sm font-semibold text-gray-700 mb-2">Strategy Genome</h4>
            <div class="bg-gray-50 rounded p-3 text-xs font-mono text-gray-800 whitespace-pre-wrap break-words">
${escapeHtml(cell.dsl_genome)}
            </div>
        </div>

        <!-- Phenotypes -->
        ${phenotypes.length > 0 ? `
        <div class="mb-4">
            <h4 class="text-sm font-semibold text-gray-700 mb-2">Multi-Timeframe Performance</h4>
            <div class="space-y-2">
                ${phenotypes.map(p => `
                    <div class="bg-gray-50 rounded p-2 text-xs">
                        <div class="flex justify-between items-center">
                            <span class="font-medium text-gray-700">${p.timeframe.toUpperCase()}</span>
                            <span class="${p.total_profit >= 0 ? 'text-green-600' : 'text-red-600'} font-medium">
                                $${p.total_profit.toFixed(2)}
                            </span>
                        </div>
                        <div class="text-gray-600 mt-1">
                            ${p.total_trades} trades
                            ${p.sharpe_ratio ? `• Sharpe: ${p.sharpe_ratio.toFixed(2)}` : ''}
                        </div>
                    </div>
                `).join('')}
            </div>
        </div>
        ` : ''}

        <!-- Lineage -->
        <div class="mb-4">
            <h4 class="text-sm font-semibold text-gray-700 mb-2">Lineage (${lineage.length} ancestors)</h4>
            <div class="lineage-tree bg-gray-50 rounded p-3 text-xs">
                ${renderLineageText(lineage)}
            </div>
        </div>

        ${cell.llm_name ? `
        <!-- LLM Analysis -->
        <div class="mb-4">
            <h4 class="text-sm font-semibold text-gray-700 mb-2">LLM Analysis</h4>
            <div class="bg-blue-50 rounded p-3 text-xs">
                <div class="font-medium text-blue-900">${cell.llm_name}</div>
                ${cell.llm_category ? `<div class="text-blue-700">Category: ${cell.llm_category}</div>` : ''}
                ${cell.llm_hypothesis ? `<div class="text-blue-700 mt-1">${escapeHtml(cell.llm_hypothesis)}</div>` : ''}
            </div>
        </div>
        ` : ''}
    `;
}

// Render lineage tree as text (for inline display in cell details panel)
function renderLineageText(lineage) {
    return lineage.map((ancestor, index) => {
        const indent = '  '.repeat(index);
        const arrow = index > 0 ? '↓\n' + indent : '';
        const fitnessChange = index > 0
            ? ` (${(ancestor.fitness - lineage[index - 1].fitness >= 0 ? '+' : '')}$${(ancestor.fitness - lineage[index - 1].fitness).toFixed(2)})`
            : '';

        return `${arrow}Gen ${ancestor.generation}: Cell #${ancestor.cell_id} → $${ancestor.fitness.toFixed(2)}${fitnessChange}`;
    }).join('\n');
}

// Load dish info
async function loadDishInfo(dishName) {
    try {
        const response = await fetch(`${API_BASE}/dish/${dishName}`);
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

        const dish = await response.json();

        // Update dish info bar
        document.getElementById('dish-info-name').textContent = dish.dish_name;
        document.getElementById('dish-info-description').textContent = dish.description || 'No description';
        document.getElementById('dish-info-generations').textContent = dish.total_generations;
        document.getElementById('dish-info-runs').textContent = dish.total_runs;
        document.getElementById('dish-info-symbol').textContent = dish.symbol;

        // Show dish info bar
        document.getElementById('dish-info-bar').classList.remove('hidden');

        console.log(`Loaded dish info for: ${dishName}`);
    } catch (error) {
        console.error('Failed to load dish info:', error);
        logActivity('error', `Failed to load dish info: ${error.message}`);
    }
}

// Setup event listeners
function setupEventListeners() {
    // Dish selector change
    document.getElementById('dish-selector').addEventListener('change', async (e) => {
        selectedDish = e.target.value || null;
        console.log(`Dish selected: ${selectedDish || 'Most Recent'}`);
        logActivity('system', `Switched to dish: ${selectedDish || 'Most Recent'}`);

        // Load dish info if a dish is selected
        if (selectedDish) {
            await loadDishInfo(selectedDish);
        } else {
            // Hide dish info bar
            document.getElementById('dish-info-bar').classList.add('hidden');
        }

        await loadCells(currentFilter);
    });

    // Cell row click
    document.getElementById('cell-table-body').addEventListener('click', (e) => {
        const row = e.target.closest('.cell-row');
        if (row) {
            const cellId = parseInt(row.dataset.cellId);

            // Update selection styling
            document.querySelectorAll('.cell-row').forEach(r => r.classList.remove('selected'));
            row.classList.add('selected');

            // Load cell details
            loadCellDetails(cellId);
        }
    });

    // Run button - Start trading-learn
    document.getElementById('run-btn').addEventListener('click', async () => {
        try {
            const dishMsg = selectedDish ? ` on dish "${selectedDish}"` : '';
            logActivity('system', `Starting trading-learn (30 iterations, local LLM)${dishMsg}...`);

            const requestBody = {
                iterations: 30
            };
            if (selectedDish) {
                requestBody.dish = selectedDish;
            }

            const response = await fetch(`${API_BASE}/learn/run`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestBody)
            });

            const result = await response.json();

            if (result.status === 'success') {
                logActivity('success', `✓ ${result.message} (PID: ${result.pid})`);

                // Update button states
                document.getElementById('run-btn').disabled = true;
                document.getElementById('run-btn').classList.add('opacity-50', 'cursor-not-allowed');
                document.getElementById('stop-btn').disabled = false;
                document.getElementById('stop-btn').classList.remove('opacity-50', 'cursor-not-allowed');

                // Update status dot to green
                document.getElementById('activity-status-dot').classList.remove('bg-gray-500', 'bg-red-500');
                document.getElementById('activity-status-dot').classList.add('bg-green-500');

                // Start polling for output
                startOutputPolling();
            } else {
                logActivity('error', `✗ ${result.message}`);
            }
        } catch (error) {
            logActivity('error', `✗ Failed to start: ${error.message}`);
        }
    });

    // Stop button - Stop trading-learn
    document.getElementById('stop-btn').addEventListener('click', async () => {
        try {
            logActivity('system', 'Stopping trading-learn process...');

            const response = await fetch(`${API_BASE}/learn/stop`, {
                method: 'POST'
            });

            const result = await response.json();

            if (result.status === 'success') {
                logActivity('success', `✓ ${result.message}`);

                // Update button states
                document.getElementById('run-btn').disabled = false;
                document.getElementById('run-btn').classList.remove('opacity-50', 'cursor-not-allowed');
                document.getElementById('stop-btn').disabled = true;
                document.getElementById('stop-btn').classList.add('opacity-50', 'cursor-not-allowed');

                // Update status dot to gray
                document.getElementById('activity-status-dot').classList.remove('bg-green-500');
                document.getElementById('activity-status-dot').classList.add('bg-gray-500');

                // Stop polling
                stopOutputPolling();
            } else {
                logActivity('error', `✗ ${result.message}`);
            }
        } catch (error) {
            logActivity('error', `✗ Failed to stop: ${error.message}`);
        }
    });

    // Clear activity log
    document.getElementById('clear-log').addEventListener('click', () => {
        activityLog = [];
        updateActivityLog();
        logActivity('system', 'Log cleared');
    });

    // Open activity log in new window
    document.getElementById('fullscreen-log').addEventListener('click', () => {
        const logContent = document.getElementById('activity-log').innerHTML;
        const newWindow = window.open('', 'Trading-Learn Output', 'width=1200,height=800');

        newWindow.document.write(`
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Trading-Learn Output</title>
                <script src="https://cdn.tailwindcss.com"></script>
                <style>
                    body {
                        font-family: 'Courier New', monospace;
                        background: #0d1117;
                        color: #c9d1d9;
                    }
                </style>
            </head>
            <body class="p-6">
                <div class="flex justify-between items-center mb-4 pb-4 border-b border-gray-700">
                    <h1 class="text-2xl font-bold text-white">Trading-Learn Live Output</h1>
                    <div class="flex items-center gap-3">
                        <div id="status-indicator" class="flex items-center gap-2">
                            <div class="w-3 h-3 rounded-full ${isPollingActive ? 'bg-green-500' : 'bg-gray-500'} animate-pulse"></div>
                            <span class="text-sm">${isPollingActive ? 'Running' : 'Stopped'}</span>
                        </div>
                        <button onclick="window.close()" class="px-3 py-1 bg-red-600 text-white text-sm rounded hover:bg-red-700">
                            Close
                        </button>
                    </div>
                </div>
                <div id="output-container" class="bg-gray-900 rounded-lg p-4 max-h-screen overflow-y-auto">
                    ${logContent}
                </div>
                <script>
                    // Auto-refresh output every 2 seconds
                    setInterval(() => {
                        fetch('${API_BASE}/learn/output?lines=100')
                            .then(res => res.json())
                            .then(data => {
                                if (data.output) {
                                    document.getElementById('output-container').innerHTML = formatOutput(data.output);
                                    document.getElementById('output-container').scrollTop = document.getElementById('output-container').scrollHeight;
                                }
                            })
                            .catch(err => console.error('Error fetching output:', err));
                    }, 2000);

                    function formatOutput(rawOutput) {
                        const lines = rawOutput.split('\\n');
                        return lines
                            .filter(line => line.trim().length > 0)
                            .map(line => {
                                if (line.includes('✓') || line.includes('SUCCESS')) {
                                    return '<div class="text-green-400 mb-1">✓ ' + escapeHtml(line.replace(/✓/g, '').trim()) + '</div>';
                                } else if (line.includes('✗') || line.includes('ERROR') || line.includes('Failed')) {
                                    return '<div class="text-red-400 mb-1">✗ ' + escapeHtml(line.replace(/✗/g, '').trim()) + '</div>';
                                } else if (line.includes('Analyzing') || line.includes('Testing') || line.includes('Creating')) {
                                    return '<div class="text-yellow-400 mb-1">⚙ ' + escapeHtml(line.trim()) + '</div>';
                                } else if (line.includes('Cell') || line.includes('Strategy') || line.includes('Fitness')) {
                                    return '<div class="text-blue-400 mb-1">📊 ' + escapeHtml(line.trim()) + '</div>';
                                } else if (line.includes('Iteration') || line.includes('Phase')) {
                                    return '<div class="text-purple-400 mb-1 font-semibold">' + escapeHtml(line.trim()) + '</div>';
                                } else {
                                    return '<div class="text-gray-400 mb-1">' + escapeHtml(line.trim()) + '</div>';
                                }
                            })
                            .join('');
                    }

                    function escapeHtml(text) {
                        const div = document.createElement('div');
                        div.textContent = text;
                        return div.innerHTML;
                    }
                </script>
            </body>
            </html>
        `);
        newWindow.document.close();
    });

    // Filter buttons
    document.getElementById('filter-all').addEventListener('click', () => {
        currentFilter = null;
        updateFilterButtons('filter-all');
        loadCells(null);
    });

    document.getElementById('filter-llm').addEventListener('click', () => {
        currentFilter = true;
        updateFilterButtons('filter-llm');
        loadCells(true);
    });

    document.getElementById('filter-evolution').addEventListener('click', () => {
        currentFilter = false;
        updateFilterButtons('filter-evolution');
        loadCells(false);
    });
}

// Output polling for trading-learn process
let outputPollingInterval = null;
let isPollingActive = false;
let apiLoggingEnabled = true;  // Control API logging in fetch wrapper

function formatTradingLearnOutput(rawOutput) {
    // Parse and format the output nicely
    const lines = rawOutput.split('\n');
    const formattedLines = lines
        .filter(line => line.trim().length > 0)
        .map(line => {
            // Color code different types of messages
            if (line.includes('✓') || line.includes('SUCCESS')) {
                return `<div class="text-green-400 mb-1">✓ ${escapeHtml(line.replace(/✓/g, '').trim())}</div>`;
            } else if (line.includes('✗') || line.includes('ERROR') || line.includes('Failed')) {
                return `<div class="text-red-400 mb-1">✗ ${escapeHtml(line.replace(/✗/g, '').trim())}</div>`;
            } else if (line.includes('Analyzing') || line.includes('Testing') || line.includes('Creating')) {
                return `<div class="text-yellow-400 mb-1">⚙ ${escapeHtml(line.trim())}</div>`;
            } else if (line.includes('Cell') || line.includes('Strategy') || line.includes('Fitness')) {
                return `<div class="text-blue-400 mb-1">📊 ${escapeHtml(line.trim())}</div>`;
            } else if (line.includes('Iteration') || line.includes('Phase')) {
                return `<div class="text-purple-400 mb-1 font-semibold">${escapeHtml(line.trim())}</div>`;
            } else {
                return `<div class="text-gray-400 mb-1">${escapeHtml(line.trim())}</div>`;
            }
        })
        .join('');

    return formattedLines || '<div class="text-gray-500">Waiting for output...</div>';
}

function startOutputPolling() {
    isPollingActive = true;
    apiLoggingEnabled = false;  // Disable API logging during learn output display

    // Poll every 2 seconds for new output
    outputPollingInterval = setInterval(async () => {
        try {
            // Use raw fetch to bypass API monitoring
            const response = await window.originalFetch(`${API_BASE}/learn/output?lines=50`);
            const data = await response.json();

            if (data.output) {
                const logEl = document.getElementById('activity-log');
                logEl.innerHTML = formatTradingLearnOutput(data.output);
                // Auto-scroll to bottom
                logEl.scrollTop = logEl.scrollHeight;
            }
        } catch (error) {
            console.error('Failed to fetch output:', error);
        }
    }, 2000);
}

function stopOutputPolling() {
    isPollingActive = false;
    apiLoggingEnabled = true;  // Re-enable API logging when learn stops
    if (outputPollingInterval) {
        clearInterval(outputPollingInterval);
        outputPollingInterval = null;
    }
}

// Update filter button styles
function updateFilterButtons(activeId) {
    const buttons = ['filter-all', 'filter-llm', 'filter-evolution'];
    buttons.forEach(id => {
        const btn = document.getElementById(id);
        if (id === activeId) {
            btn.classList.remove('bg-gray-100', 'text-gray-700');
            btn.classList.add('bg-blue-100', 'text-blue-700');
        } else {
            btn.classList.remove('bg-blue-100', 'text-blue-700');
            btn.classList.add('bg-gray-100', 'text-gray-700');
        }
    });
}

// Utility: Escape HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// WebSocket connection for real-time events
let eventWebSocket = null;

function connectEventWebSocket() {
    // WebSocket is on API server (port 8081), not static file server (port 8082)
    const wsUrl = `ws://localhost:8081/ws/events`;

    logActivity('system', `Connecting to event stream at ${wsUrl}...`);

    eventWebSocket = new WebSocket(wsUrl);

    eventWebSocket.onopen = () => {
        logActivity('success', '✓ Connected to real-time event stream');
    };

    eventWebSocket.onmessage = (event) => {
        try {
            const eventData = JSON.parse(event.data);

            // Map event types to log categories
            const typeMap = {
                'CELL_ANALYSIS_START': 'llm',
                'CELL_ANALYSIS_PROGRESS': 'llm',
                'PATTERN_DISCOVERED': 'llm',
                'MUTATION_PROPOSED': 'llm',
                'MUTATION_TESTED': 'evolution',
                'CELL_BIRTHED': 'success',
                'APPLICATION_WARNING': 'error',
                'APPLICATION_ERROR': 'error'
            };

            const category = typeMap[eventData.type] || 'system';
            const content = eventData.content || JSON.stringify(eventData);

            logActivity(category, content);
        } catch (err) {
            console.error('Error processing WebSocket message:', err);
        }
    };

    eventWebSocket.onerror = (error) => {
        console.error('WebSocket error:', error);
        logActivity('error', '✗ Event stream connection error');
    };

    eventWebSocket.onclose = () => {
        logActivity('system', 'Event stream disconnected. Reconnecting in 5s...');
        setTimeout(connectEventWebSocket, 5000);
    };
}

// Start the app
document.addEventListener('DOMContentLoaded', () => {
    init();
    connectEventWebSocket();
});
