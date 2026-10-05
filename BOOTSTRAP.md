# HAOps Science — Research Workspace Bootstrap Guide

> **Platform:** HAOps Science
> **Purpose:** First-time setup guide for your AI research agent workspace
> **Read this:** Before starting your first Claude Code research session

---

## Prerequisites

Install these **before** starting:

### Required
```bash
# Claude Code (AI agent) — follow instructions at https://claude.com/claude-code
# Verify: claude --version

# Node.js 18+ (for MCP server)
# Verify: node --version

# Git
# Verify: git --version

# HAOps Science MCP Server — clone and build
git clone https://github.com/tbranzov/haops-science-mcp-server.git
cd haops-science-mcp-server
npm install && npm run build
```

### Recommended
```bash
# Python 3 (data analysis, visualizations)
# Verify: python3 --version
pip3 install matplotlib seaborn pandas    # common analysis libraries

# Marp CLI (presentations → PDF/PPTX/HTML)
npm install -g @marp-team/marp-cli
# Verify: marp --version
```

### Optional
```bash
# LaTeX (local PDF compilation — server can also compile)
# macOS:
brew install --cask mactex
# Linux (texlive-lang-cyrillic is REQUIRED — it provides t2aenc.def for the T2A
# fontenc + Bulgarian/Russian babel the assembler emits for Cyrillic papers;
# without it, any paper containing Cyrillic fails with
# "Encoding file 't2aenc.def' not found"):
sudo apt install texlive-latex-base texlive-latex-extra texlive-fonts-recommended \
  texlive-lang-cyrillic texlive-lang-european biber
# Verify: pdflatex --version
# Verify Cyrillic support: kpsewhich t2aenc.def   (must print a path)
```

---

## Quick Setup (3 minutes)

### Step 1: Extract ZIP

Download and extract the workspace ZIP from HAOps Science:

```bash
# Extract the downloaded ZIP
unzip Проект на Никола-workspace.zip -d /tmp/research-workspace
cd /tmp/research-workspace
```

### Step 2: Run setup script

```bash
# Make executable and run
chmod +x scripts/setup.sh
./scripts/setup.sh /path/to/your/project
```

The setup script copies:
- `.claude/` → `/path/to/your/project/.claude/` (agent definitions + settings)
- `memory/` → `~/.claude/projects/.../memory/` (research agent memory)
- `CLAUDE.md` → `/path/to/your/project/CLAUDE.md` (project context)
- `private/PROJECT-INFO.md` → `/path/to/your/project/private/PROJECT-INFO.md` (credentials)

### Step 3: Fill in Research Topic

Edit `CLAUDE.md` and complete the **Research Topic** section:

```bash
# Open in your editor
code /path/to/your/project/CLAUDE.md
# or: nano, vim, etc.
```

Fill in:
- **Main Research Question** — the central question your research answers
- **Scope** — boundaries of the investigation
- **Objectives** — 2-4 concrete deliverables

### Step 4: Register MCP server

The HAOps MCP server gives your agent access to the Knowledge Graph, References Library, and paper management tools.

```bash
# Use --scope user so the server is available across all your projects
cd /path/to/your/project
claude mcp add-json --scope user haops '{"type":"stdio","command":"node","args":["/path/to/haops-science-mcp-server/dist/index.js"],"env":{"HAOPS_API_URL":"https://science.haops.eu","HAOPS_API_KEY":"YOUR_API_KEY"}}'

# Verify connection
claude mcp list
# Expected: haops - ✓ Connected
```

> **API Key:** Your API key is included in `private/PROJECT-INFO.md`. If missing, generate one in HAOps Science → Settings → API Keys.

### Step 5: Start your first session

```bash
cd /path/to/your/project
claude
```

Tell Claude your role:

```
"You are the research agent for this project. Read your memory and boot sequence."
```

The agent will read `memory/research-agent.md` → `CLAUDE.md` → `research-agent-patterns.md` → `research-agent-history.md`, then prompt you for first tasks.

---

## First Session Goals

Once your agent is running, suggested first tasks:

1. **Update protocol** — The auto-created protocol may be outdated. Tell the agent:
   "Read the latest protocol from the repo template and update it with haops_update_protocol"
2. **Seed the Knowledge Graph** — Create 3-5 foundational nodes (hypothesis, method, concept)
3. **Import core references** — Use DOI lookup or BibTeX import for your top 10 sources
4. **Create paper outline** — Scaffold your paper structure with standard sections

See `memory/research-agent-history.md` for the Getting Started guide with example MCP calls.

---

## Verification Checklist

After setup, confirm:

- [ ] `CLAUDE.md` exists and Research Topic is filled in
- [ ] `private/PROJECT-INFO.md` has your API key
- [ ] `memory/` directory populated with research-agent files
- [ ] `.claude/agents/` has `research-agent.md`
- [ ] `claude mcp list` shows `haops - ✓ Connected`
- [ ] First Claude session boots without placeholder errors

---

## Troubleshooting

### MCP won't connect

**Symptom:** `claude mcp list` shows `haops - ✗ Error`

**Solutions:**
1. Verify the MCP server path: `node /path/to/haops-science-mcp-server/dist/index.js` should run without errors
2. Check your API key is valid — test with: `curl -H "Authorization: Bearer YOUR_KEY" https://science.haops.eu/api/projects`
3. Re-register: `claude mcp remove haops` then re-run the `claude mcp add-json` command

### Placeholder issues in CLAUDE.md

**Symptom:** Agent sees `Проект на Никола` or `[PROJECT_NAME]` literally

**Cause:** ZIP was generated with missing project metadata (researchType, discipline, citationStyle not set in project settings).

**Solution:** Set project metadata in HAOps Science → Project Settings → Research Metadata, then re-download the workspace ZIP.

### Agent doesn't understand the project

**Symptom:** Agent asks "what is this project about?" or gives generic responses

**Solution:** The Research Topic section in `CLAUDE.md` is empty or too vague. Expand it:
- Add concrete research questions
- Describe the domain and scope
- List 2-3 specific objectives

### Missing `haops_*` tools in Claude

**Symptom:** Agent says it doesn't have `haops_read_memory` or similar tools

**Solution:** MCP registration was skipped. Re-run Step 4. Restart Claude Code after registration: `exit` then `claude` again.

### Setup script permission denied

```bash
# Make executable manually
chmod +x scripts/setup.sh
bash scripts/setup.sh /path/to/your/project
```

---

## Manual Bootstrap (without setup.sh)

If the script fails, copy files manually:

```bash
PROJECT=/path/to/your/project
MEMORY_DIR=~/.claude/projects/$(echo $PROJECT | tr '/' '-' | sed 's/^-//')/memory

# Create directories
mkdir -p "$PROJECT/.claude/agents"
mkdir -p "$PROJECT/private"
mkdir -p "$MEMORY_DIR"

# Copy files
cp -r .claude/agents/ "$PROJECT/.claude/agents/"
cp .claude/settings.json "$PROJECT/.claude/settings.json"
cp -r memory/ "$MEMORY_DIR/"
cp CLAUDE.md "$PROJECT/CLAUDE.md"
cp private/PROJECT-INFO.md "$PROJECT/private/PROJECT-INFO.md"
```

---

**Generated by HAOps Science Onboarding**
**Version:** 2.0.0
