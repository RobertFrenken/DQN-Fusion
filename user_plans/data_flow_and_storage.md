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

- I think what might be the best of both worlds is to store data and the database on my home directory
in rf15. Here I have 500GB of potential storage. The issue is that I am the only one who can read and
write to this storage. I have asked my advisor to increase the project storage to 1 TB on PAS1266.
This way when my undergraduate students finally get onboarded, they can also read and permissively
write to my data lake. I might need to add some permissions to the data lake so others with project
access can't change it by accident.

- After reading osc documentation, it also seems prudent to utilize the scratch space, as it has
much faster read and write speeds for jobs. I will need to develop a process of long term storage
in the "core" spot (first home then project), and each ML training job will write the needed data
to scratch, will train and write its outputs (model weights, training metrics, etc) on scratch, and
then cleanup can be to transfer those files from scratch to the data lake.
- One note is that I also have configured wandb cloud for real time analysis and quick dashboards
of my training progress. I will need to research if I can gain that benefit of keeping data
on osc and allowing some sort of link to the wandb web browser, or if I just stream tracking metrics
to their cloud in addition to writing to my data lake. I am fine with either decision.
- The next note is handling any training errors during runs. This can include poor code and logic,
or if I hit some slurm error like OOM, walltime, etc. How do half jobs get handled in the data lake
and on wandb?

- Finally, I want there to many "plug ins" or connections to the datalake. This can include post-hoc
analysis of training metrics, playground instances of pulling and prototyping models and visuals,
and a connection to my front facing webpage of my results. The website is currently a static site,
but I think I will commit to either using react or hugging face platform for smoother and more dynamic
pipeline and allow for interactive queries on the app client side.

