/**
 * Mosaic coordinator init + Parquet loading helpers.
 * Import in OJS cells: import { vg, loadParquetTable } from "./_ojs/mosaic-setup.js"
 */

const vg = await import("https://cdn.jsdelivr.net/npm/@uwdata/vgplot@0.21.1/+esm");

// Initialize DuckDB-WASM coordinator (singleton)
await vg.coordinator().databaseConnector(vg.wasmConnector());

/**
 * Load a Parquet file (via FileAttachment URL) into a DuckDB table.
 * @param {string} tableName - DuckDB table name
 * @param {string} url - URL to Parquet file (from FileAttachment.url())
 */
async function loadParquetTable(tableName, url) {
  await vg.coordinator().exec(
    vg.loadParquet(tableName, url)
  );
}

export { vg, loadParquetTable };
