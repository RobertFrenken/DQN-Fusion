/**
 * Mosaic coordinator init + Parquet loading helpers.
 *
 * Usage in OJS cells:
 *   import { vg, loadParquetTable, listTables, describeTable } from "./_ojs/mosaic-setup.js"
 *
 * IMPORTANT: wasmConnector() returns synchronously but DuckDB-WASM initializes
 * lazily on first query. We must call getDuckDB() to force full initialization
 * before passing to databaseConnector(), otherwise internal objects aren't ready
 * and you get "t.addEventListener is not a function".
 */

async function initMosaic() {
  const mod = await import("https://cdn.jsdelivr.net/npm/@uwdata/vgplot@0.21.1/+esm");
  const wasm = mod.wasmConnector();
  await wasm.getDuckDB();
  mod.coordinator().databaseConnector(wasm);
  return mod;
}

const vg = await initMosaic();

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

/**
 * List all tables currently loaded in DuckDB-WASM.
 * @returns {Promise<Array<{name: string}>>} Array of table objects
 */
async function listTables() {
  const result = await vg.coordinator().query(
    vg.sql`SHOW TABLES`
  );
  return Array.from(result);
}

/**
 * Describe a table's columns.
 * @param {string} tableName - DuckDB table name
 * @returns {Promise<Array<{column_name: string, column_type: string}>>}
 */
async function describeTable(tableName) {
  const result = await vg.coordinator().query(
    `DESCRIBE ${tableName}`
  );
  return Array.from(result);
}

export { vg, loadParquetTable, listTables, describeTable };
