/**
 * CellList component - displays evolved trading strategy cells
 */

import { Component } from "../core.js";

export class CellList extends Component {
  constructor() {
    super();
    this.attachShadow({ mode: "open" });

    // Add styles
    const style = document.createElement("style");
    style.textContent = `
      :host {
        display: block;
        color: white;
      }
      .header {
        margin-bottom: 1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
      }
      h2 {
        font-size: 1.25rem;
        font-weight: 600;
        margin: 0;
      }
      .count {
        font-size: 0.875rem;
        color: #9ca3af;
      }
      table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.875rem;
      }
      thead {
        background: #374151;
        position: sticky;
        top: 0;
      }
      th {
        padding: 0.75rem;
        text-align: left;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.75rem;
        letter-spacing: 0.05em;
        color: #d1d5db;
      }
      td {
        padding: 0.75rem;
        border-bottom: 1px solid #374151;
      }
      tr:hover {
        background: #1f2937;
      }
      .fitness {
        font-weight: 600;
        font-family: monospace;
      }
      .fitness.positive {
        color: #10b981;
      }
      .fitness.zero {
        color: #9ca3af;
      }
      .fitness.negative {
        color: #ef4444;
      }
      .genome {
        font-family: monospace;
        font-size: 0.75rem;
        color: #d1d5db;
        max-width: 400px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      }
      .status {
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.75rem;
        font-weight: 500;
      }
      .status.online {
        background: #10b98144;
        color: #10b981;
      }
      .status.deprecated {
        background: #6b728044;
        color: #9ca3af;
      }
      .error {
        color: #ef4444;
        padding: 1rem;
        text-align: center;
      }
      .loading {
        color: #9ca3af;
        padding: 1rem;
        text-align: center;
      }
      .empty {
        color: #9ca3af;
        padding: 2rem;
        text-align: center;
      }
    `;
    this.shadowRoot.appendChild(style);

    // Create container
    this.container = document.createElement("div");
    this.shadowRoot.appendChild(this.container);

    // Fetch cells data
    this.fetchCells();

    // Refresh every 5 seconds
    this.refreshInterval = setInterval(() => this.fetchCells(), 5000);
  }

  disconnectedCallback() {
    if (this.refreshInterval) {
      clearInterval(this.refreshInterval);
    }
  }

  async fetchCells() {
    try {
      const response = await fetch("/api/cells?limit=100");
      const data = await response.json();
      this.setState({ cells: data.cells, total_count: data.total_count, error: null, loading: false });
    } catch (error) {
      console.error("Error fetching cells:", error);
      this.setState({ error: error.message, loading: false });
    }
  }

  getFitnessClass(fitness) {
    if (fitness > 0) return "positive";
    if (fitness < 0) return "negative";
    return "zero";
  }

  render() {
    if (this.state.loading) {
      this.container.innerHTML = '<div class="loading">Loading cells...</div>';
      return;
    }

    if (this.state.error) {
      this.container.innerHTML = `<div class="error">Error: ${this.state.error}</div>`;
      return;
    }

    const cells = this.state.cells || [];
    const totalCount = this.state.total_count || 0;

    if (cells.length === 0) {
      this.container.innerHTML = `
        <div class="empty">
          <p>No cells found</p>
          <p style="font-size: 0.75rem; margin-top: 0.5rem;">Run trading-evolve to create cells</p>
        </div>
      `;
      return;
    }

    this.container.innerHTML = `
      <div class="header">
        <h2>Evolution Cells</h2>
        <span class="count">${cells.length} of ${totalCount} cells</span>
      </div>
      <table>
        <thead>
          <tr>
            <th>ID</th>
            <th>Gen</th>
            <th>Fitness</th>
            <th>Strategy (DSL Genome)</th>
            <th>Status</th>
            <th>Parent</th>
          </tr>
        </thead>
        <tbody>
          ${cells.map(cell => `
            <tr>
              <td>${cell.cell_id}</td>
              <td>${cell.generation}</td>
              <td class="fitness ${this.getFitnessClass(cell.fitness)}">$${cell.fitness.toFixed(2)}</td>
              <td class="genome" title="${cell.dsl_genome}">${cell.dsl_genome}</td>
              <td><span class="status ${cell.status}">${cell.status}</span></td>
              <td>${cell.parent_cell_id || '-'}</td>
            </tr>
          `).join('')}
        </tbody>
      </table>
    `;
  }
}

customElements.define("cell-list", CellList);
