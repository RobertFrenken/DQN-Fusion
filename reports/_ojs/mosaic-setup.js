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

const CDN_URL = "https://cdn.jsdelivr.net/npm/@uwdata/vgplot@0.21.1/+esm";

async function initMosaic() {
  console.log(`[mosaic-setup] Loading vgplot from ${CDN_URL}`);
  const mod = await import(CDN_URL);
  console.log("[mosaic-setup] vgplot module loaded, initializing DuckDB-WASM...");
  const wasm = mod.wasmConnector();
  await wasm.getDuckDB();
  console.log("[mosaic-setup] DuckDB-WASM initialized, connecting coordinator...");
  mod.coordinator().databaseConnector(wasm);
  console.log("[mosaic-setup] Mosaic ready");
  return mod;
}

let vg;
try {
  vg = await initMosaic();
} catch (error) {
  console.error("[mosaic-setup] Failed to initialize Mosaic/vgplot:", error);

  // Create a proxy that throws a descriptive error on any property access,
  // so downstream OJS cells get a useful message instead of "vg is not defined".
  const handler = {
    get(_, prop) {
      if (prop === Symbol.toPrimitive || prop === "toString" || prop === "valueOf") {
        return () => "[mosaic-setup: initialization failed]";
      }
      throw new Error(
        `[mosaic-setup] Cannot use vg.${String(prop)}() — Mosaic initialization failed.\n` +
        `Root cause: ${error.message}\n` +
        `Check: (1) CDN reachable? (2) Page served over HTTP, not file://? ` +
        `(3) Browser console for network errors.`
      );
    }
  };
  vg = new Proxy({}, handler);
}

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
