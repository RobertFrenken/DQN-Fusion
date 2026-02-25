/**
 * Reactive data fetching for Svelte 5 islands.
 *
 * Usage:
 *   let emb = resource<EmbeddingPoint[]>(
 *     () => `/data/embeddings/${run}_vgae_${method}.json`
 *   );
 *   // In template: {emb.data}, {emb.loading}, {emb.error}
 *
 * Automatically refetches when the URL changes (tracked via $effect).
 * Aborts in-flight requests on URL change or component teardown.
 * Unwraps the standard {schema_version, exported_at, data} envelope.
 */

interface Envelope<T> {
  schema_version: string;
  exported_at: string;
  data: T;
}

interface ResourceState<T> {
  readonly data: T | undefined;
  readonly loading: boolean;
  readonly error: string | null;
}

export function resource<T>(urlFn: () => string): ResourceState<T> {
  let data = $state<T | undefined>();
  let loading = $state(false);
  let error = $state<string | null>(null);

  $effect(() => {
    const url = urlFn(); // tracked: refetch when URL changes
    loading = true;
    error = null;
    const ctrl = new AbortController();

    fetch(url, { signal: ctrl.signal })
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
        return r.json();
      })
      .then((raw: Envelope<T> | T) => {
        // Unwrap envelope if present
        data = (raw as Envelope<T>).data ?? (raw as T);
        loading = false;
      })
      .catch((e) => {
        if (e.name !== 'AbortError') {
          error = e.message;
          loading = false;
        }
      });

    return () => ctrl.abort();
  });

  return {
    get data() { return data; },
    get loading() { return loading; },
    get error() { return error; },
  };
}
