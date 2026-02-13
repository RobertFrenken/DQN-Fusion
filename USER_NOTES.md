# User Investigation Questions

## Memory Batch Size Issues
Does pytorch geometric, pytorch, or any other resource have an accurate means of
actually correctly prediciting the appropiate batch size for a model run?
Currently Pytorch Lightning Tuner is not sufficient and we need a clunky
workaround with a "safety factor" for each dataset. For 6 datasets this might
work, but as we increase the datasets this will become a configuration nightmare.

## Config files and their locations
After much trial and error, it seems that the ideal place for "magic numbers"
or parameters that will be changed is to create a configuration setup. Though 
this has helped with tracking, there are currently many config files of different
types (JSON, YAML, in python, etc) in different locations. In my estimation this
seems fragile at best, and may cause headaches when needing to scale up.

A subquestion to this is the number of configurations and the seperability of them.
Currently there are a couple of huge configuration spots that mass dump configs.
There is little inheritance and though might be okay for claude, is not human
readable or understandable.

A question I have is having a single snakemake file bad? It seems to get larger
and larger and it might become fragile.

## "Flat" Project Folders
There are currently many project folders under root, and certain processes
need to "climb" up the directory and back down to reach certain files and 
configs. There should be more careful planning into trying to create a "top down"
approach, we need to discuss the pros and cons of each.

## Staleness issues
Claude files must constantly be manually refreshed after each change. For a
single person eyeing the context it is managable, but long term there needs
to be a way for continual improvement or an awareness for claude to update
itself and its context.

## Skepticism on Claude's ability to find best practices
Claude has a tendency to assume what is currently present is immovable, and
must be continually prompted to ensure that we are not going down a design
path that we will regret in the future. A prime example was the detour of 
fragile folder setups to track models and runs, when a registry was a simple fix
that is common in every workflow solution. That cost a ton of precious time and
resources.

Looking online, I found this R package for pipelines: https://mlr3book.mlr-org.com/
There are sections of pipelines and advanced techniques that at first glance
might be useful.

OSC + SLURM also have some native calls and tooling that we don't seem to use
for tracking, monitoring, or allocating resources. They may not be fruitful,
but I would have liked claude to at least check them.

Also, wandb seems to have some interesting tracking abilities:
- https://wandb.ai/site/artifacts/
- https://wandb.ai/site/registry/
- https://wandb.ai/site/articles/what-is-an-ml-model-registry/
- https://wandb.ai/site/reports/
- https://wandb.ai/site/sdk/
- https://wandb.ai/site/sweeps/ (later)
- https://wandb.ai/site/weave/
- https://wandb.ai/site/ruler/

Some blogs on how top labs do research:
- https://www.lesswrong.com/posts/6P8GYb4AjtPXx6LLB/tips-and-code-for-empirical-research-workflows

## Claude's eagerness to be complex
Adjacent to the above section, claude likes to jump into very intricate fixes
for a particular problem, instead of really asking "this problem has likely
been encountered before, so there is likely a solution I can find". This is
similar to how claude writes code. It would rather creates hundreds of lines
of code instead of trying to create elegant or consise solutions. This has
a doubly negative side effect. The code is more fragile, and claude now bloats
precious context trying to store and conceptualize all the poor code it wrote,
which makes each additional intervention worse.

## Versioning and package management
Currently use miniconda as the package manager and python interpreter. Perhaps
this is okay, but Claude always has trouble calling it for its own tests.
I keep hearing about uv. Maybe it won't work here, but I am constantly worried
that miniconda is fragile.

## Claudes dual dotfiles
There are a ton of dotfiles about the project root of KD-GAT. Some of them seem
okay like .conda or .ssh, but I am worried that a 2nd set of .claude files might
cause confusion and context bloat.

## Connection to my Mkdocs lab github
I am trying to make a context-rich documentation both for people, and perhaps
even for llm agents (https://osu-car-msl.github.io/lab-setup-guide/). If claude
has trouble with context, is there a way to use this site as a private lookup
database? Can claude look at my papers repo to help populate values, or fill in
new findings?

## Ability to log and query open research questions
Is there a way to separate the software engineering + MLOps part of the code
to asking particular research questions. For instance, currently the framework
does very poorly with OOD datasets. This is a research question and not an operations
question, so I want it logged but not actively called upon if we are in
"MLops" mode. Other questions are the potential of using JumpRelu for adversarial
training, cascading knowledge distillation, etc. I want claude to toggle between
what kinds of contexts or what attention it should have depending on our session.

This is similar to the "meta prompting" or steering problem.
- https://github.com/gsd-build/get-shit-done

## Claude to suggest out of the box ideas
There are many things I don't know, but claude assumes that my prompt is the
best thing ever and there can't be a better idea. That isn't true, I want claude
to suggest designs or ideas I haven't even considered that could really help.

