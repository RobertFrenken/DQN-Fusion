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

## 2. Shell Upgrade: bash vs zsh vs fish

### Option A: Stay on bash + add starship prompt (Recommended Starting Point)

**Migration effort**: Minimal (one line in `.bashrc`)
**Risk**: Zero -- all existing scripts, conda, module commands keep working

bash 5.1.8 on OSC is modern enough. The main thing it lacks out of the box is a good prompt. Install starship (single binary, no root needed):

```bash
# Install to ~/.local/bin
curl -sS https://starship.rs/install.sh | sh -s -- --bin-dir ~/.local/bin

# Add to ~/.bashrc
eval "$(~/.local/bin/starship init bash)"
```

Starship shows: current directory, git branch/status, conda env, Python version, command duration, exit codes. All configurable via `~/.config/starship.toml`.

### Option B: Switch to zsh + oh-my-zsh (More Features, More Setup)

**Migration effort**: Medium -- most bash syntax works in zsh, but some edge cases differ
**Risk**: Low -- zsh is in `/etc/shells` on OSC, so you can set it as your login shell

**What zsh gains over bash:**
- Better tab completion (e.g., `git ch<TAB>` shows `checkout`, `cherry-pick`)
- Shared history across sessions
- Spelling correction
- Glob patterns (`**/*.py` works natively)
- Right-side prompt (show conda env on the right)

**oh-my-zsh** is a framework that bundles themes + plugins:
```bash
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
```

**Essential plugins for HPC/ML:**
```bash
# In ~/.zshrc
plugins=(
    git                    # Git aliases (gst=status, gco=checkout, etc.)
    zsh-autosuggestions    # Fish-style suggestions from history (grayed out text)
    zsh-syntax-highlighting # Colors commands as you type (red = not found)
    conda-zsh-completion   # Better conda tab completion
)
```

**zsh + Powerlevel10k** is the most popular theme:
```bash
git clone --depth=1 https://github.com/romkatv/powerlevel10k.git \
  ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/themes/powerlevel10k
# Set ZSH_THEME="powerlevel10k/powerlevel10k" in ~/.zshrc
# Run p10k configure for interactive setup
```

Powerlevel10k uses "instant prompt" to eliminate the startup delay that plagues oh-my-zsh. This matters for SSH sessions where shell startup latency is noticeable.

**Migration gotchas on HPC:**
- `module load` works fine in zsh (it's POSIX-compatible at the function level)
- `conda init zsh` must be run once to update `.zshrc`
- Some bash-specific syntax (e.g., `[[ $var =~ regex ]]`) may behave slightly differently
- You can always fall back: `chsh -s /bin/bash` or just type `bash` to get a bash session
- Keep `#!/bin/bash` shebangs in your scripts -- they run bash regardless of your login shell

**Startup time concern:** Plain oh-my-zsh with many plugins can add 200-500ms to shell startup. With Powerlevel10k's instant prompt and lazy-loading, this is reduced to ~50ms. On SSH this matters less since you're already absorbing network latency.

### Option C: fish (Not Recommended for HPC)

fish has the best out-of-box experience (autosuggestions, syntax highlighting, web-based config) but it is **not POSIX-compatible**. This means:
- `module load` may not work correctly (uses bash/sh functions)
- conda init produces fish-specific config, but integration is fragile
- Snakemake shell commands assume bash
- Many HPC tutorials and scripts assume bash/POSIX syntax

fish is great for local development machines. On HPC, stick with bash or zsh.

### Recommendation

**Phase 1 (now)**: bash + starship prompt. Zero risk, immediate quality-of-life improvement.
**Phase 2 (when comfortable)**: Try zsh in a tmux pane -- `zsh` to enter, `exit` to leave. If you like it, set it as your login shell with `chsh -s /bin/zsh`.

---

## 3. Terminal Emulators (Local Machine)

This section is about what runs on **your laptop/desktop** -- the application you type into. Since you SSH into OSC, the terminal emulator affects rendering speed, font quality, and local features like tabs/splits.

### Current Setup: VS Code Integrated Terminal

**Pros:**
- No separate app needed -- terminal is right next to your code
- Remote SSH extension handles connection management
- Shared clipboard, file navigation, Git GUI
- Extensions for everything

**Cons:**
- Terminal performance is mediocre (Electron-based, not GPU-accelerated)
- Large output (build logs, training output) can lag
- If VS Code crashes, your terminal sessions die (unless using tmux on the remote)
- Limited terminal customization (fonts, colors, key bindings)

**Verdict:** VS Code's terminal is fine for quick commands. For long-running sessions, SSH into a tmux session from a dedicated terminal.

### GPU-Accelerated Terminals

All of these are significantly faster than VS Code's terminal for rendering large amounts of output (training logs, data processing).

#### Kitty

- **Config**: Plain text file (`~/.config/kitty/kitty.conf`)
- **Splits/tabs**: Built-in, no tmux needed for basic layouts
- **SSH integration**: `kitten ssh` command copies your local config (fonts, colors) to the remote
- **Image support**: Kitty graphics protocol (display images inline in terminal)
- **Platform**: Linux, macOS (not Windows)
- **Font rendering**: Excellent, GPU-accelerated
- **Best for**: Power users who want deep customization and use Linux/Mac

#### WezTerm

- **Config**: Lua scripting (`~/.config/wezterm/wezterm.lua`)
- **Splits/tabs**: Built-in multiplexer (can replace tmux for local work)
- **SSH integration**: Built-in SSH client with multiplexing -- can connect to remotes directly
- **Image support**: Sixel + iTerm2 protocol + Kitty protocol
- **Platform**: Linux, macOS, **Windows**
- **Font rendering**: Excellent, GPU-accelerated
- **Best for**: Cross-platform users, people who want an all-in-one terminal

#### Alacritty

- **Config**: TOML file (`~/.config/alacritty/alacritty.toml`)
- **Splits/tabs**: None -- by design, you use tmux for this
- **SSH integration**: None -- just a terminal, you run `ssh` manually
- **Image support**: Sixel (added recently)
- **Platform**: Linux, macOS, Windows
- **Font rendering**: Good, GPU-accelerated
- **Best for**: Minimalists who use tmux for everything

#### Ghostty

- **Config**: Simple key-value text file
- **Splits/tabs**: Built-in with native platform integration
- **SSH integration**: None built-in
- **Image support**: Kitty graphics protocol
- **Platform**: Linux, macOS (native UI on both -- uses GTK on Linux, AppKit on macOS)
- **Font rendering**: Best-in-class, written by HashiCorp's founder
- **Best for**: People who want "just works" native performance with minimal config

#### Windows Terminal (Windows only)

- **Config**: JSON
- **Splits/tabs**: Built-in
- **SSH integration**: Works with OpenSSH, WSL
- **Platform**: Windows
- **Best for**: Windows users who need WSL integration

#### iTerm2 (macOS only)

- **Config**: GUI preferences
- **Splits/tabs**: Built-in
- **SSH integration**: `it2ssh` for tmux integration (attach remote tmux as native tabs)
- **Image support**: iTerm2 inline images protocol
- **Platform**: macOS only
- **Best for**: Mac users who want a polished GUI terminal

### Which Terminal Should You Use?

**If you're on macOS**: Ghostty or Kitty. Both are fast and have great font rendering. Ghostty is newer and more "zero-config". Kitty is more mature with better SSH integration (`kitten ssh`).

**If you're on Windows**: WezTerm. Cross-platform, great built-in multiplexer, configurable via Lua.

**If you're on Linux**: Kitty or Ghostty. Both are GPU-accelerated with native Linux support.

**Regardless**: Keep using VS Code for editing/browsing code. Use a dedicated terminal for SSH sessions where you run tmux + Snakemake + Claude Code.

### External Terminal + SSH vs VS Code Remote SSH

| Aspect | VS Code Remote SSH | External Terminal + SSH |
|--------|-------------------|----------------------|
| Code editing | Excellent (full IDE) | Not applicable (use VS Code for this) |
| Terminal performance | Mediocre | Fast (GPU-accelerated) |
| Session persistence | Dies with VS Code | Survives in tmux |
| Resource usage | Heavy (VS Code server on login node) | Lightweight |
| Git integration | GUI + terminal | Terminal only |
| File browsing | GUI sidebar | `ls`, `find`, etc. |

**Best approach**: Use both. VS Code for editing/Git GUI. External terminal for long-running tmux sessions (Snakemake, Claude Code, monitoring).

---

## 4. CLI Tools (User-Installable)

None of these are installed system-wide on OSC, but all can be installed to `~/.local/bin/` without root access.

### High Impact

**fzf** -- Fuzzy finder for everything
```bash
git clone --depth 1 https://github.com/junegunn/fzf.git ~/.fzf
~/.fzf/install
```
- `Ctrl-r` -- fuzzy search command history (replaces bash's reverse search)
- `Ctrl-t` -- fuzzy find files
- `Alt-c` -- fuzzy cd into directories
- Pipe anything into it: `squeue -u rf15 | fzf` to filter jobs interactively

**lazygit** -- Terminal Git UI
```bash
# Download binary to ~/.local/bin
LAZYGIT_VERSION=$(curl -s "https://api.github.com/repos/jesseduffield/lazygit/releases/latest" | grep -Po '"tag_name": "v\K[^"]*')
curl -Lo lazygit.tar.gz "https://github.com/jesseduffield/lazygit/releases/latest/download/lazygit_${LAZYGIT_VERSION}_Linux_x86_64.tar.gz"
tar xf lazygit.tar.gz lazygit && mv lazygit ~/.local/bin/
```
- Full Git GUI in the terminal: staging hunks, interactive rebase, branch management
- Useful when VS Code Git extensions have issues (which you mentioned)
- Far more intuitive than raw git commands for complex operations (cherry-pick, rebase, conflict resolution)

### Medium Impact

**bat** -- `cat` with syntax highlighting and line numbers
```bash
# Download from GitHub releases to ~/.local/bin
```
- Replaces `cat` for reading files: `bat pipeline/stages.py`
- Integrates with fzf for previews

**eza** -- Modern `ls` replacement
```bash
# Download from GitHub releases to ~/.local/bin
```
- `eza -la` -- colored, icons, git status per file
- `eza --tree --level=2` -- tree view

**ripgrep** (`rg`) -- Fast grep
```bash
# Download from GitHub releases to ~/.local/bin
```
- Already familiar from Claude Code (which bundles it)
- `rg "def train_fusion" --type py` -- searches project instantly

**zoxide** -- Smarter `cd`
```bash
# Download from GitHub releases to ~/.local/bin
```
- Learns your most-used directories
- `z kd-gat` jumps to `/users/PAS2022/rf15/CAN-Graph-Test/KD-GAT` from anywhere

**btop** -- Better htop
```bash
# Download from GitHub releases to ~/.local/bin
```
- GPU monitoring (useful for checking V100 utilization)
- More visual than htop

### Installation Pattern

All these tools are single static binaries. The pattern is always:
1. Download the `linux_x86_64` release from GitHub
2. Extract to `~/.local/bin/`
3. Add `export PATH="$HOME/.local/bin:$PATH"` to your `.bashrc` (probably already there)

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
