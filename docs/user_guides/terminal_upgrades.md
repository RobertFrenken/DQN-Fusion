# Terminal & Workflow Upgrades for HPC Development

**Updated**: 2026-02-02
**Context**: OSC Pitzer cluster (RHEL 9), VS Code Remote SSH, ML pipeline work

## What's Available on OSC Right Now

Checked on `pitzer-login04`:

| Tool | Version | Status |
|------|---------|--------|
| bash | 5.1.8 | Current default shell |
| zsh | 5.8 | Available, in `/etc/shells` (can set as login shell) |
| fish | installed | Available, in `/etc/shells` |
| tmux | 3.2a | Available |
| screen | 4.08.00 | Available |
| htop | installed | Available |
| fzf, bat, eza, starship, lazygit, btop | -- | Not installed (user-installable) |

---

## 1. tmux (Priority: High -- Start Here)

tmux is already installed and is the single highest-impact upgrade for HPC work.

### Why It Matters

Any process tied to your SSH session dies when the connection drops. This includes:
- Snakemake orchestrating a DAG of SLURM jobs
- Claude Code CLI sessions (long context, mid-task)
- Large file transfers (`rsync`, `scp`)
- Interactive debugging sessions

tmux decouples your processes from the SSH connection.

### Essential Commands

```
tmux new -s pipeline          # Create named session
tmux new -s claude            # Another named session
tmux ls                       # List all sessions
tmux attach -t pipeline       # Reattach to session
tmux kill-session -t pipeline # Destroy session
```

**Inside tmux:**

| Key | Action |
|-----|--------|
| `Ctrl-b d` | Detach (session keeps running) |
| `Ctrl-b %` | Split pane vertically |
| `Ctrl-b "` | Split pane horizontally |
| `Ctrl-b arrow` | Move between panes |
| `Ctrl-b c` | New window (like a tab) |
| `Ctrl-b n` / `Ctrl-b p` | Next / previous window |
| `Ctrl-b [` | Enter scroll mode (q to exit) |
| `Ctrl-b z` | Zoom/unzoom current pane |

### Recommended `~/.tmux.conf`

```bash
# Better prefix (Ctrl-a is closer to home row than Ctrl-b)
unbind C-b
set -g prefix C-a
bind C-a send-prefix

# Mouse support (scroll, click panes, resize)
set -g mouse on

# Start window numbering at 1 (0 is far from prefix key)
set -g base-index 1
setw -g pane-base-index 1

# Bigger scrollback buffer (default is 2000)
set -g history-limit 50000

# Intuitive split keys
bind | split-window -h -c "#{pane_current_path}"
bind - split-window -v -c "#{pane_current_path}"

# Reload config
bind r source-file ~/.tmux.conf \; display "Config reloaded"

# Don't rename windows automatically
set -g allow-rename off

# Reduce escape delay (important for vim/neovim users)
set -sg escape-time 10

# 256 color support
set -g default-terminal "screen-256color"
```

### HPC-Specific Workflow Patterns

**Pattern 1: Pipeline monitoring**
```bash
tmux new -s pipeline
# Pane 1: snakemake running
# Ctrl-a | to split
# Pane 2: watch -n 30 squeue -u rf15
# Ctrl-a - to split pane 2 horizontally
# Pane 3: tail -f experimentruns/hcrl_sa/teacher_fusion/slurm.err
```

**Pattern 2: Claude Code**
```bash
tmux new -s claude
claude   # Start Claude Code CLI
# Detach with Ctrl-a d when done for the day
# Reattach later: tmux attach -t claude
```

**Pattern 3: Data transfer**
```bash
tmux new -s transfer
rsync -avP large_dataset/ /fs/scratch/PAS1266/data/
# Detach -- transfer continues even if you close your laptop
```

### tmux Plugins (Optional)

Install tmux plugin manager (tpm):
```bash
git clone https://github.com/tmux-plugins/tpm ~/.tmux/plugins/tpm
```

Add to `~/.tmux.conf`:
```bash
# Plugins
set -g @plugin 'tmux-plugins/tpm'
set -g @plugin 'tmux-plugins/tmux-sensible'      # Better defaults
set -g @plugin 'tmux-plugins/tmux-resurrect'      # Save/restore sessions (Ctrl-a Ctrl-s / Ctrl-a Ctrl-r)
set -g @plugin 'tmux-plugins/tmux-continuum'      # Auto-save every 15 min

# Initialize (keep at bottom)
run '~/.tmux/plugins/tpm/tpm'
```

Then press `Ctrl-a I` (capital I) inside tmux to install plugins.

**tmux-resurrect** is particularly useful on HPC: if the login node reboots (rare but happens during maintenance), you can restore your window layout with `Ctrl-a Ctrl-r`. Note: it restores layout and working directories, not running processes.

### tmux vs screen

Both are available on OSC. tmux is the better choice:
- Better split-pane support (screen's is awkward)
- Mouse support built-in
- Plugin ecosystem (tpm)
- Active development (screen is effectively in maintenance mode)
- tmux uses a server model (can have multiple clients attach to the same session)

### HPC Caveats

- **Login node reboots**: tmux sessions live on the login node. If it reboots, sessions are lost (but SLURM jobs keep running). This is rare on OSC but does happen during maintenance. `tmux-resurrect` mitigates this.
- **Multiple login nodes**: `ssh pitzer.osc.edu` may land you on different login nodes (login01-04). Your tmux session is on the specific node. Use `ssh pitzer-login04.hpc.osc.edu` to return to the same node, or just check `tmux ls` after connecting.
- **conda/module**: `conda activate` and `module load` work fine inside tmux panes. Each pane is an independent shell.

---

## 2. Shell: bash + starship (Recommended)

bash 5.1.8 on OSC is modern enough. Add starship for a better prompt (git branch, conda env, command duration):

```bash
# Install to ~/.local/bin (no root needed)
curl -sS https://starship.rs/install.sh | sh -s -- --bin-dir ~/.local/bin

# Add to ~/.bashrc
eval "$(~/.local/bin/starship init bash)"
```

Configure via `~/.config/starship.toml`. zsh is also available on OSC (`/etc/shells`) if you want autosuggestions and better tab completion, but bash + starship + fzf gets you 80% of the way with zero migration risk.

---

## 3. Terminal Emulators (Local Machine)

VS Code's integrated terminal is fine for quick commands but lags on large output (training logs). For long-running SSH sessions, use a GPU-accelerated terminal alongside VS Code:

- **macOS**: Ghostty (zero-config) or Kitty (`kitten ssh` copies config to remote)
- **Windows**: WezTerm (built-in multiplexer, Lua config)
- **Linux**: Kitty or Ghostty

Use VS Code for editing. Use a dedicated terminal for tmux sessions (Snakemake, Claude Code, monitoring).

---

## 4. CLI Tools (User-Installable)

Not installed system-wide on OSC. All are single static binaries — download the `linux_x86_64` release from GitHub, extract to `~/.local/bin/`, ensure `$PATH` includes it.

| Tool | What it does |
|------|-------------|
| **fzf** | Fuzzy finder — `Ctrl-r` history search, `Ctrl-t` file finder, pipe anything into it |
| **lazygit** | Terminal Git UI — staging hunks, rebase, branch management |
| **bat** | `cat` with syntax highlighting and line numbers |
| **eza** | Modern `ls` — colored output, git status, tree view |
| **ripgrep** (`rg`) | Fast grep (Claude Code bundles it) |
| **zoxide** | Smarter `cd` — `z kd-gat` from anywhere |
| **btop** | Better htop with GPU monitoring |

```bash
# Example: install fzf
git clone --depth 1 https://github.com/junegunn/fzf.git ~/.fzf && ~/.fzf/install
```

---

## 5. Recommended Setup Order

Priority order based on impact vs effort:

1. **tmux** -- Already installed. Create `~/.tmux.conf` with the config above. Start using it for Snakemake and Claude Code immediately.

2. **fzf** -- Quick install, transforms `Ctrl-r` history search. Pays for itself in the first session.

3. **starship prompt** -- Single binary, one line in `.bashrc`. Shows git branch, conda env, last command duration.

4. **lazygit** -- Single binary. Solves the "VS Code git extensions have issues" problem directly.

5. **External terminal** (Kitty/WezTerm/Ghostty) -- Install on your local machine for SSH sessions. Use alongside VS Code, not instead of it.

6. **zsh + oh-my-zsh** (optional) -- Only if you want the autosuggestions and syntax highlighting. Starship + fzf on bash gets you 80% of the way.

7. **bat, eza, ripgrep, zoxide, btop** -- Nice-to-haves. Install when you find yourself wanting them.

---

## 6. Quick Reference: Day-to-Day HPC Workflow

```
# Start of work session
ssh pitzer-login04.hpc.osc.edu       # Always same login node for tmux
tmux attach -t pipeline || tmux new -s pipeline

# In tmux pane 1: run pipeline
conda activate gnn-experiments
snakemake -s pipeline/Snakefile --profile profiles/slurm --config 'datasets=["hcrl_sa"]'

# Ctrl-a | (split pane)
# In tmux pane 2: monitor
watch -n 30 squeue -u rf15

# Ctrl-a c (new window)
# In tmux window 2: Claude Code
tmux rename-window claude
claude

# Detach when done: Ctrl-a d
# Close laptop, go home, reconnect later
```
