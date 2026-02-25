import { defineCollection } from "astro:content";
import { file } from "astro/loaders";
import { z } from "astro/zod";

// Envelope parser factory: unwrap {schema_version, exported_at, data} → data[]
// and inject unique `id` field using a key function.
function unwrapWithId<T extends Record<string, unknown>>(
  keyFn: (item: T, index: number) => string,
) {
  return (text: string) => {
    const parsed = JSON.parse(text);
    const data: T[] = Array.isArray(parsed.data) ? parsed.data : [parsed.data];
    return data.map((item, i) => ({ id: keyFn(item, i), ...item }));
  };
}

const leaderboard = defineCollection({
  loader: file(
    "src/data/leaderboard.json",
    {
      parser: unwrapWithId((d) =>
        `${d.dataset}_${d.model_type}_${d.scale}_${d.has_kd}_${d.model}_${d.metric_name}`,
      ),
    },
  ),
  schema: z.object({
    dataset: z.string(),
    model_type: z.string(),
    scale: z.string(),
    has_kd: z.number(),
    model: z.string(),
    metric_name: z.string(),
    best_value: z.number(),
  }),
});

const runs = defineCollection({
  loader: file(
    "src/data/runs.json",
    { parser: unwrapWithId((d) => String(d.run_id)) },
  ),
  schema: z.object({
    run_id: z.string(),
    dataset: z.string(),
    model_type: z.string(),
    scale: z.string(),
    stage: z.string(),
    has_kd: z.number(),
    status: z.string(),
    teacher_run: z.string(),
    started_at: z.string(),
    completed_at: z.string(),
  }),
});

const datasets = defineCollection({
  loader: file(
    "src/data/datasets.json",
    { parser: unwrapWithId((d) => String(d.name)) },
  ),
  schema: z.object({
    name: z.string(),
    domain: z.string(),
    protocol: z.string(),
    source: z.string(),
    description: z.string(),
  }),
});

const kdTransfer = defineCollection({
  loader: file(
    "src/data/kd_transfer.json",
    {
      parser: unwrapWithId((d) =>
        `${d.student_run}_${d.metric_name}`,
      ),
    },
  ),
  schema: z.object({
    student_run: z.string(),
    dataset: z.string(),
    model_type: z.string(),
    student_scale: z.string(),
    teacher_run: z.string(),
    metric_name: z.string(),
    student_value: z.number(),
    teacher_value: z.number(),
  }),
});

const modelSizes = defineCollection({
  loader: file(
    "src/data/model_sizes.json",
    { parser: unwrapWithId((d) => `${d.model_type}_${d.scale}`) },
  ),
  schema: z.object({
    model_type: z.string(),
    scale: z.string(),
    param_count: z.number(),
    param_count_M: z.number(),
  }),
});

export const collections = {
  leaderboard,
  runs,
  datasets,
  kdTransfer,
  modelSizes,
};
