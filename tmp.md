Alright, honest thoughts. Some of these are sharp observations, some I think are misdirected.          
  The "Claude's eagerness to be complex" section (line 67-74) is the most important thing in this document, and you've nailed the diagnosis:       
  overengineered code bloats context, which degrades future output, which creates a compounding quality spiral. That's a real dynamic and it's good
   you're aware of it.

  The config proliferation problem (line 10-24) is also well-identified. You have JSON, YAML, Python dataclasses, and Snakemake configs all doing
  overlapping jobs. That's genuine tech debt.

  The batch size complaint (line 3-8) is valid. Lightning's tuner is known to be unreliable with PyG's irregular batch sizes. There's no magic
  bullet here — the "safety factor" hack is ugly but it's what most people actually do. You could look into PyG's max_num_nodes-based estimation,
  but it's still heuristic.

  Where I'd push back:

  The W&B links (line 53-59) worry me. You already have MLflow + SQLite + Datasette + Snakemake tracking. That's three-and-a-half tracking systems.
   Your CLAUDE.md calls this "triple storage" like it's an architecture decision, but it reads more like accretion. Adding W&B would be a fourth
  system, not a fix. The actual problem is consolidation, not adoption of another tool. If you're unhappy with MLflow, replace it — don't layer on
  top.

  The mlr3 R package reference (line 45) — I'm not sure what you'd extract from an R ML framework for a PyTorch/PyG pipeline. The pipeline concepts
   (DAGs, caching, checkpointing) are universal, but you already have Snakemake doing that. This feels like browsing for solutions rather than
  diagnosing the specific problem.

  The "flat project folders" section (line 26-29) is vague. You say there should be "more careful planning" but don't say what's actually breaking.
   Directory restructuring mid-project is expensive and disruptive. Unless you have a concrete pain point (circular imports? path resolution
  failures?), this is cosmetic and not worth the churn.

  The things that are real but harder than you think:

  "Ability to log and query open research questions" (line 94-103) — the desire to toggle between MLOps mode and research mode is a genuinely
  interesting idea. But this is essentially asking for a session-scoped attention mechanism on top of an LLM. Right now the practical version of
  this is just well-organized CLAUDE.md files and being explicit at the start of a session about what mode you're in. The get-shit-done link is a
  prompt-engineering repo, not a system that solves this architecturally.

  "Claude to suggest out of the box ideas" (line 107-109) — this is in tension with your (correct) complaint about Claude being too complex. If you
   want Claude to freelance on architecture, you'll get more complexity, not less. The more productive framing is: "Claude should push back when my
   request implies unnecessary complexity" rather than "Claude should suggest novel designs." Those are opposite failure modes.

  The elephant in the room:

  Most of these notes are symptoms of a single tension: this project is simultaneously a research prototype and an attempt at production-grade
  MLOps infrastructure. Research wants to move fast, break things, and explore. MLOps wants reproducibility, tracking, and rigid pipelines. You're
  building both at once with one person, and the friction you're feeling is that tension — not any individual tool choice.

  If I were you, I'd pick one: either lock down the pipeline and stop adding new tracking tools, or accept that the infrastructure is messy and
  focus on getting research results. Trying to do both at once is why you have four config formats and three tracking systems.
  ⎿  Tool use rejected with user message: The user wants to clarify these questions.                                                               
         This means they may have additional information, context or questions for you.                                                            
         Take their response into account and then reformulate the questions if appropriate.
         Start by asking them what they would like to clarify.                                   
❯ The "flat project folders" is my concern about implicitly enforcing stronger information flow with my project. For instance, configs are at      
  the "top" as they are non executive files that drive the outcome of the project. The pipeline would be a step lower, and then components         
  like pytorch models, training logic, etc would be below that, as a pytorch model isn't going to "call up" a pipeline function. On the wandb,     
   I wanted to see if they are doing anything innovative or good practices that I am currently not doing. I want to have strong tracking and       
  dashboards so I thought they would have some strong design patterns to use. the "toggling" idea is an attempt to separate that during            
  different sessions I have different goals. Some are operational, while others are research driven. I wanted to know if there is a protocol       
  or way to segment these distinct modes. If most of claude's gains are context based, and that is mostly markdown, then it would stand to         
  reason there could be effectively different mini markdown databases for different types of information, and that would allow you to only         
  ingest what is needed (operational context vs research context).                                                                               
  On the layered architecture: Now I understand — you're talking about enforcing a dependency DAG, not just folder aesthetics. That's a legitimate 
  concern. Right now pipeline/ imports from src/, but configs live at multiple levels, and there's no enforced boundary preventing, say, a model   
  file from reaching up into pipeline logic. A clean layering like:                                                           
  configs (data, inert)  →  pipeline (orchestration)  →  src (components)

  ...where arrows mean "can depend on" and nothing flows backwards — that's a sound principle. The current structure partially does this, but it's
  not enforced. This would be worth doing, though I'd suggest it as a focused refactor rather than a full reorg, since moving files breaks
  Snakemake paths, import paths, and cached outputs all at once.

  On W&B as design pattern research: That reframing makes much more sense. You're right that W&B has strong design patterns worth studying even
  without adopting the tool. Their artifact lineage (tracking which model came from which dataset+config), sweep configuration (declarative
  hyperparameter search), and report templates (reproducible dashboards) are genuinely well-designed. The useful question is: which of those
  patterns are you missing in your current MLflow+SQLite setup, and can you implement them without adding a dependency? For instance, artifact
  lineage could be a few extra columns in your project DB.

  On context toggling: This is your most interesting idea, and you're reasoning about it correctly. If Claude's behavior is primarily shaped by
  what markdown it ingests, then curating different markdown "profiles" per session type is essentially manual attention management. You already
  have the infrastructure for this — your .claude/system/ directory has STATE.md, CONVENTIONS.md, PROJECT_OVERVIEW.md. The extension would be
  something like:

  - .claude/system/ops/ — pipeline status, infrastructure concerns, tracking patterns, current bugs
  - .claude/system/research/ — open questions (OOD generalization, JumpReLU, cascading KD), literature notes, experiment hypotheses

  Then at session start, you'd indicate which mode, and only the relevant context gets loaded. Claude Code doesn't natively support conditional
  context loading today, but there are practical approximations — like a MODE.md file you swap, or structuring your CLAUDE.md to reference
  different files depending on the session goal.

  The deeper version of this idea connects to your MkDocs site question too — if you had structured markdown knowledge bases (one for ops, one for
  research, one for lab onboarding), Claude could fetch from them as needed rather than front-loading everything into context.