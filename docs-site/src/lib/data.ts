/**
 * Typed fetch helpers for Tier 2 (per-run) data.
 * These load JSON from public/data/ at runtime in Svelte islands.
 */

const BASE = "/data";

interface Envelope<T> {
  schema_version: string;
  exported_at: string;
  data: T;
}

async function loadJSON<T>(
  path: string,
  signal?: AbortSignal,
): Promise<T> {
  const res = await fetch(`${BASE}/${path}`, { signal });
  if (!res.ok) throw new Error(`Failed to load ${path}: ${res.status}`);
  const raw: Envelope<T> = await res.json();
  return raw.data;
}

// --- Per-run data loaders ---

export interface TrainingPoint {
  epoch: number;
  [metric: string]: number;
}

export const loadTrainingCurves = (runId: string, signal?: AbortSignal) =>
  loadJSON<TrainingPoint[]>(`training_curves/${runId}.json`, signal);

export interface EmbeddingPoint {
  dim0: number;
  dim1: number;
  label: number | string;
  [key: string]: unknown;
}

export const loadEmbeddings = (
  runId: string,
  model: string,
  method: string,
  signal?: AbortSignal,
) =>
  loadJSON<EmbeddingPoint[]>(
    `embeddings/${runId}_${model}_${method}.json`,
    signal,
  );

export interface AttentionData {
  heads: number[][];
  labels: string[];
  [key: string]: unknown;
}

export const loadAttention = (runId: string, signal?: AbortSignal) =>
  loadJSON<AttentionData>(`attention/${runId}.json`, signal);

export interface DqnPolicyData {
  alpha_values: Record<string, number>;
  [key: string]: unknown;
}

export const loadDqnPolicy = (runId: string, signal?: AbortSignal) =>
  loadJSON<DqnPolicyData>(`dqn_policy/${runId}.json`, signal);

export interface ReconErrorData {
  errors: number[];
  labels: (number | string)[];
  [key: string]: unknown;
}

export const loadReconErrors = (runId: string, signal?: AbortSignal) =>
  loadJSON<ReconErrorData>(`recon_errors/${runId}.json`, signal);

export interface RocCurveData {
  fpr: number[];
  tpr: number[];
  auc: number;
  [key: string]: unknown;
}

export const loadRocCurves = (
  runId: string,
  model: string,
  signal?: AbortSignal,
) =>
  loadJSON<RocCurveData>(`roc_curves/${runId}_${model}.json`, signal);

export interface CkaData {
  similarity_matrix: number[][];
  layer_names: string[];
  [key: string]: unknown;
}

export const loadCka = (runId: string, signal?: AbortSignal) =>
  loadJSON<CkaData>(`cka/${runId}.json`, signal);

export interface MetricData {
  [metric: string]: number;
}

export const loadMetrics = (runId: string, signal?: AbortSignal) =>
  loadJSON<MetricData>(`metrics/${runId}.json`, signal);

export interface GraphSampleData {
  nodes: Array<{ id: string; [key: string]: unknown }>;
  links: Array<{ source: string; target: string; [key: string]: unknown }>;
  [key: string]: unknown;
}

export const loadGraphSamples = (signal?: AbortSignal) =>
  loadJSON<GraphSampleData>("graph_samples.json", signal);
