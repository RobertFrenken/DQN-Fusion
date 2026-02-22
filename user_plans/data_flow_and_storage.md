## Problem
The current state of the project is a hectic state where data is in many different locations,
both within osc, aws, wandb, and perhaps other locations. This sprawl is fragile at best, and
at worst will cause continual crashes.

## Current Ideas
- one of the current roadblocks is the storage constraints on osc itself. I have plenty of storage
in my home directory, where I have 500GB of allocation. Each individual project (PAS####) only has
100GB of storage, and it might be shared among every other osc member. There is also at least 1TB
of storage on the scratch project folders, but that data is wiped after 60 days, which makes it
an infesible location for a long term database.
- This has prompted the move to an aws s3 bucket where storage is abundant. There is native integration
with osc, but this might cause issues with reading and writing. there is also the issue of syncing,
and the data transfers from OSC to s3, and wandb to s3. Then everything must be exported and synced
in s3 for the static dashboard website to pull all the appropiate data over. Again, at best this is
fragile and at worst will constantly break.

- Another issue faced earlier was the race to write to sqllite db during the pipeline, as only 1 job
could write at a time, causing lockout. Snakemake had trouble handling this, which has lead to refactoring
with prefect's slurm declaration headers. This is a good start, but after more research we will be
moving to the Ray ecosystem.

