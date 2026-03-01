## Summary

`wasmConnector()` returns a Promise in v0.18.0+ (after the connector class refactor), but the [Get Started documentation](https://idl.uw.edu/mosaic/get-started/) shows synchronous usage without `await`:

```javascript
vg.coordinator().databaseConnector(vg.wasmConnector());
```

This causes `TypeError: t.addEventListener is not a function` because `databaseConnector()` receives an unresolved Promise instead of the actual connector object.

## Reproduction

**Environment:** Quarto 1.6 OJS cells with `@uwdata/vgplot@0.21.1` via jsDelivr CDN

```javascript
// Broken — passes Promise to databaseConnector()
vg = {
  const mod = await import("https://cdn.jsdelivr.net/npm/@uwdata/vgplot@0.21.1/+esm");
  await mod.coordinator().databaseConnector(mod.wasmConnector());
  return mod;
}
```

**Error in browser console:**
```
TypeError: t.addEventListener is not a function
```

All downstream cells that reference `vg` then fail with `vg is not defined`.

## Working Fix

Await `wasmConnector()` separately before passing to `databaseConnector()`:

```javascript
vg = {
  const mod = await import("https://cdn.jsdelivr.net/npm/@uwdata/vgplot@0.21.1/+esm");
  const wasm = await mod.wasmConnector();
  mod.coordinator().databaseConnector(wasm);
  return mod;
}
```

This matches the pattern used in [cscheid/quarto-dashboard-ojs-examples](https://github.com/cscheid/quarto-dashboard-ojs-examples/blob/main/flights.qmd) (albeit with v0.4.0).

## Suggested Doc Fix

The [Get Started page](https://idl.uw.edu/mosaic/get-started/) and [API connector docs](https://idl.uw.edu/mosaic/api/core/connectors/) should show `wasmConnector()` as async:

```diff
- vg.coordinator().databaseConnector(vg.wasmConnector());
+ const wasm = await vg.wasmConnector();
+ vg.coordinator().databaseConnector(wasm);
```

## Notes

- Issue #845 also shows the synchronous pattern and was not flagged as the cause
- The v0.18.0 changelog mentions "expose database connector classes, deprecate initializer methods" but doesn't document the async behavior change
- The [API docs for `wasmConnector`](https://idl.uw.edu/mosaic/api/core/connectors/) describe parameters but don't mention the return type is a Promise
