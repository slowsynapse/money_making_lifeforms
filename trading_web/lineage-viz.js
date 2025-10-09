// D3.js Lineage Tree Visualization
// Creates beautiful animated evolutionary tree showing cell ancestry

function renderLineageTree(lineage) {
    const container = document.getElementById('lineage-viz');
    const section = document.getElementById('lineage-section');

    // Show the section
    section.style.display = 'block';

    // Clear previous visualization
    container.innerHTML = '';

    if (!lineage || lineage.length === 0) {
        container.innerHTML = '<p class="text-gray-500 text-center py-8">No lineage data available</p>';
        return;
    }

    // Set up dimensions - responsive height based on number of nodes
    const width = container.clientWidth;
    const baseHeight = 300;
    const heightPerNode = 50;
    const height = Math.max(baseHeight, Math.min(500, baseHeight + (lineage.length * heightPerNode)));
    const nodeRadius = 30;
    const nodeSpacing = Math.min(150, width / (lineage.length + 1));

    // Create SVG
    const svg = d3.select(container)
        .append('svg')
        .attr('width', width)
        .attr('height', height);

    // Create gradient definitions for fancy effects
    const defs = svg.append('defs');

    // Gradient for positive fitness
    const gradientGreen = defs.append('linearGradient')
        .attr('id', 'gradient-green')
        .attr('x1', '0%').attr('y1', '0%')
        .attr('x2', '0%').attr('y2', '100%');
    gradientGreen.append('stop').attr('offset', '0%').attr('stop-color', '#10b981');
    gradientGreen.append('stop').attr('offset', '100%').attr('stop-color', '#059669');

    // Gradient for negative fitness
    const gradientRed = defs.append('linearGradient')
        .attr('id', 'gradient-red')
        .attr('x1', '0%').attr('y1', '0%')
        .attr('x2', '0%').attr('y2', '100%');
    gradientRed.append('stop').attr('offset', '0%').attr('stop-color', '#f87171');
    gradientRed.append('stop').attr('offset', '100%').attr('stop-color', '#dc2626');

    // Glow filter for selected node
    const filter = defs.append('filter')
        .attr('id', 'glow')
        .attr('x', '-50%').attr('y', '-50%')
        .attr('width', '200%').attr('height', '200%');
    filter.append('feGaussianBlur').attr('stdDeviation', '4').attr('result', 'coloredBlur');
    const feMerge = filter.append('feMerge');
    feMerge.append('feMergeNode').attr('in', 'coloredBlur');
    feMerge.append('feMergeNode').attr('in', 'SourceGraphic');

    // Prepare data with positions
    const nodes = lineage.map((cell, i) => ({
        ...cell,
        x: (width / 2) - ((lineage.length - 1) * nodeSpacing / 2) + (i * nodeSpacing),
        y: height / 2,
        index: i
    }));

    // Create links between nodes
    const links = [];
    for (let i = 0; i < nodes.length - 1; i++) {
        links.push({
            source: nodes[i],
            target: nodes[i + 1],
            fitnessDelta: nodes[i + 1].fitness - nodes[i].fitness
        });
    }

    // Draw curved links with animated path
    const linkGroup = svg.append('g').attr('class', 'links');

    const link = linkGroup.selectAll('.lineage-link')
        .data(links)
        .enter()
        .append('path')
        .attr('class', 'lineage-link')
        .attr('d', d => {
            const sourceX = d.source.x;
            const sourceY = d.source.y;
            const targetX = d.target.x;
            const targetY = d.target.y;
            const midX = (sourceX + targetX) / 2;

            // Beautiful curved path
            return `M ${sourceX},${sourceY}
                    Q ${midX},${sourceY - 50} ${targetX},${targetY}`;
        })
        .style('stroke', d => d.fitnessDelta >= 0 ? '#10b981' : '#f87171')
        .style('stroke-opacity', 0)
        .transition()
        .duration(800)
        .delay((d, i) => i * 200)
        .style('stroke-opacity', 0.6);

    // Add fitness delta labels on links
    linkGroup.selectAll('.fitness-delta')
        .data(links)
        .enter()
        .append('text')
        .attr('class', 'fitness-delta')
        .attr('x', d => (d.source.x + d.target.x) / 2)
        .attr('y', d => d.source.y - 40)
        .attr('text-anchor', 'middle')
        .style('opacity', 0)
        .text(d => {
            const delta = d.fitnessDelta;
            return delta >= 0 ? `+$${delta.toFixed(2)}` : `-$${Math.abs(delta).toFixed(2)}`;
        })
        .style('fill', d => d.fitnessDelta >= 0 ? '#059669' : '#dc2626')
        .transition()
        .duration(500)
        .delay((d, i) => i * 200 + 400)
        .style('opacity', 1);

    // Draw nodes with animation
    const nodeGroup = svg.append('g').attr('class', 'nodes');

    const node = nodeGroup.selectAll('.lineage-node')
        .data(nodes)
        .enter()
        .append('g')
        .attr('class', 'lineage-node')
        .attr('transform', d => `translate(${d.x},${d.y})`)
        .style('opacity', 0);

    // Animate nodes appearing
    node.transition()
        .duration(600)
        .delay((d, i) => i * 200)
        .style('opacity', 1);

    // Add circles with gradient fill
    node.append('circle')
        .attr('r', 0)
        .attr('fill', d => d.fitness >= 0 ? 'url(#gradient-green)' : 'url(#gradient-red)')
        .style('filter', (d, i) => i === nodes.length - 1 ? 'url(#glow)' : 'none')
        .transition()
        .duration(500)
        .delay((d, i) => i * 200 + 200)
        .attr('r', nodeRadius);

    // Add cell ID labels
    node.append('text')
        .attr('class', 'lineage-label')
        .attr('dy', '-5')
        .text(d => `#${d.cell_id}`)
        .style('opacity', 0)
        .transition()
        .duration(400)
        .delay((d, i) => i * 200 + 400)
        .style('opacity', 1);

    // Add generation labels
    node.append('text')
        .attr('class', 'lineage-label')
        .attr('dy', '10')
        .style('font-size', '9px')
        .style('fill', '#64748b')
        .text(d => `Gen ${d.generation}`)
        .style('opacity', 0)
        .transition()
        .duration(400)
        .delay((d, i) => i * 200 + 400)
        .style('opacity', 1);

    // Add fitness values below nodes
    node.append('text')
        .attr('class', 'lineage-label')
        .attr('dy', nodeRadius + 20)
        .style('font-weight', 'bold')
        .style('fill', d => d.fitness >= 0 ? '#059669' : '#dc2626')
        .text(d => `$${d.fitness.toFixed(2)}`)
        .style('opacity', 0)
        .transition()
        .duration(400)
        .delay((d, i) => i * 200 + 600)
        .style('opacity', 1);

    // Add hover tooltips
    node.on('mouseover', function(event, d) {
        d3.select(this).select('circle')
            .transition()
            .duration(200)
            .attr('r', nodeRadius * 1.2)
            .style('filter', 'url(#glow)');

        // Show tooltip
        showTooltip(event, d);
    })
    .on('mouseout', function(event, d) {
        d3.select(this).select('circle')
            .transition()
            .duration(200)
            .attr('r', nodeRadius)
            .style('filter', d.index === nodes.length - 1 ? 'url(#glow)' : 'none');

        hideTooltip();
    })
    .on('click', function(event, d) {
        // Load detailed cell info
        loadCellDetails(d.cell_id);
    });

    // Add genome diff indicators
    if (nodes.length > 1) {
        addGenomeDiff(nodes);
    }
}

// Show tooltip on hover
function showTooltip(event, data) {
    const tooltip = d3.select('body').append('div')
        .attr('class', 'tooltip')
        .style('position', 'absolute')
        .style('background', 'white')
        .style('padding', '12px')
        .style('border-radius', '8px')
        .style('box-shadow', '0 4px 6px rgba(0,0,0,0.1)')
        .style('pointer-events', 'none')
        .style('z-index', '1000')
        .style('font-size', '12px')
        .style('max-width', '300px');

    tooltip.html(`
        <div style="font-weight: 600; margin-bottom: 4px;">Cell #${data.cell_id}</div>
        <div style="color: #64748b; font-size: 11px; margin-bottom: 8px;">Generation ${data.generation}</div>
        <div style="margin-bottom: 4px;">
            <span style="color: #64748b;">Fitness:</span>
            <span style="color: ${data.fitness >= 0 ? '#059669' : '#dc2626'}; font-weight: 600;">
                $${data.fitness.toFixed(2)}
            </span>
        </div>
        ${data.parent_cell_id ? `<div style="font-size: 11px; color: #94a3b8;">Parent: Cell #${data.parent_cell_id}</div>` : ''}
        <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid #e5e7eb; font-size: 10px; color: #9ca3af;">
            Click to view full details
        </div>
    `);

    tooltip.style('left', (event.pageX + 10) + 'px')
        .style('top', (event.pageY - 10) + 'px');
}

function hideTooltip() {
    d3.selectAll('.tooltip').remove();
}

// Add genome diff visualization to separate section
function addGenomeDiff(nodes) {
    const genomeSection = document.getElementById('genome-evolution-section');
    const container = document.getElementById('genome-diff-container');

    // Show the genome evolution section
    genomeSection.style.display = 'block';

    // Clear previous content
    container.innerHTML = '';

    for (let i = 0; i < nodes.length - 1; i++) {
        const from = nodes[i];
        const to = nodes[i + 1];

        const diffHtml = compareGenomes(from.dsl_genome, to.dsl_genome);
        const changeDiv = document.createElement('div');
        changeDiv.className = 'mb-4 pb-4 border-b border-gray-200 last:border-0';
        changeDiv.innerHTML = `
            <div class="font-medium text-gray-700 mb-2 text-sm">
                Cell #${from.cell_id} (Gen ${from.generation}) → Cell #${to.cell_id} (Gen ${to.generation})
            </div>
            <div class="genome-diff text-xs">${diffHtml}</div>
        `;
        container.appendChild(changeDiv);
    }
}

// Simple genome diff algorithm
function compareGenomes(genome1, genome2) {
    const lines1 = genome1.split('\n');
    const lines2 = genome2.split('\n');

    let result = '';
    const maxLen = Math.max(lines1.length, lines2.length);

    for (let i = 0; i < maxLen; i++) {
        const line1 = lines1[i] || '';
        const line2 = lines2[i] || '';

        if (line1 === line2) {
            result += `<span class="diff-unchanged">${escapeHtml(line2)}</span><br>`;
        } else {
            if (line1) {
                result += `<span class="diff-removed">${escapeHtml(line1)}</span><br>`;
            }
            if (line2) {
                result += `<span class="diff-added">${escapeHtml(line2)}</span><br>`;
            }
        }
    }

    return result;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
