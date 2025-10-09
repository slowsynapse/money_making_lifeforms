// Trading Evolution Dashboard
const API_BASE = 'http://localhost:8081';

let allCells = [];
let selectedCell = null;
let fitnessChart = null;

// Initialize dashboard
async function init() {
    console.log('Initializing dashboard...');
    await loadCells();
    setupEventListeners();
}

// Load cells from API
async function loadCells() {
    try {
        console.log('Fetching cells from API...');
        const response = await fetch(`${API_BASE}/cells/top/100`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        allCells = data.cells;

        console.log(`Loaded ${allCells.length} cells`);

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

        // Fetch cell details and lineage in parallel
        const [cellResponse, lineageResponse, phenotypesResponse] = await Promise.all([
            fetch(`${API_BASE}/cell/${cellId}`),
            fetch(`${API_BASE}/cell/${cellId}/lineage`),
            fetch(`${API_BASE}/cell/${cellId}/phenotypes`)
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

// Setup event listeners
function setupEventListeners() {
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
}

// Utility: Escape HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Start the app
document.addEventListener('DOMContentLoaded', init);
