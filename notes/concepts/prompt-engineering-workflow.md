# Prompt Engineering Workflow

> **Status:** Concept / Workflow Pattern
> **Related:** `notes/concepts/narrative-engine.md` (different use case)
> **Source:** `notes/research/2025-12-26/dev_console/claude_01.md`

This is about **visual behavior iteration** - testing if prompts achieve specific visual goals.

Different from narrative engine (story coherence). This is R&D for prompt/seed/LoRA effectiveness.

---

## The Use Case

"I want the prop to jiggle in place. Does this prompt do that?"

Not about story. About **visual behavior verification**:
- Does this phrasing produce the motion I want?
- Does this seed give better results?
- Does this LoRA respond to this vocabulary?
- What's the minimum prompt that achieves the effect?

---

## The Workflow: Loom-Like Branching

```
┌─────────────────────────────────────────────────────────────┐
│                                                              │
│  Generate → Step → Evaluate → Is it working?                │
│                         │                                    │
│                         ├─ Yes → Continue / Export           │
│                         │                                    │
│                         └─ No → Snapshot → Fork variations   │
│                                     │                        │
│                              ┌──────┼──────┐                │
│                              │      │      │                │
│                              ▼      ▼      ▼                │
│                           Var A  Var B  Var C               │
│                           (prompt (seed  (LoRA              │
│                            tweak) change) swap)             │
│                              │      │      │                │
│                              └──────┼──────┘                │
│                                     │                        │
│                              Compare → Pick winner           │
│                                     │                        │
│                              Continue from best              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Step-by-Step

1. **Hypothesis:** "This prompt should make the character's hand tremble"
2. **Generate:** Run a few chunks
3. **Evaluate:** Watch the output - is the hand trembling?
4. **If no:** Snapshot current state, then try variations:
   - Variation A: Different phrasing ("shaking hand" vs "trembling fingers")
   - Variation B: Different seed
   - Variation C: Add motion keywords ("subtle movement", "nervous energy")
5. **Compare:** Look at all branches side by side
6. **Select:** Pick the one that works best
7. **Continue:** Generate more from the winning branch
8. **Iterate:** Repeat until satisfied

---

## CLI Workflow

```bash
# Initial prompt
$ video-cli prompt "character holds cup, hand trembling slightly, nervous"
$ video-cli run --chunks 5

# Evaluate - is hand trembling?
$ video-cli frame --out test_frame.jpg
# (Look at it)

# Not quite right - snapshot and try variations
$ video-cli snapshot
# → snap-001

# Try variation A - different phrasing
$ video-cli prompt "character grips cup with shaky hands, visible tremor"
$ video-cli run --chunks 3
$ video-cli frame --out var_a.jpg

# Try variation B - restore and change seed
$ video-cli restore snap-001
$ video-cli seed 12345
$ video-cli run --chunks 3
$ video-cli frame --out var_b.jpg

# Try variation C - add motion emphasis
$ video-cli restore snap-001
$ video-cli prompt "character holds cup, hand trembling, micro-movements, subtle shake"
$ video-cli run --chunks 3
$ video-cli frame --out var_c.jpg

# Compare outputs, pick winner
$ video-cli restore snap-001
$ video-cli prompt "character grips cup with shaky hands, visible tremor"  # Var A won
$ video-cli run --chunks 20
```

---

## Edit-Preview Pattern

For edits that might be destructive, preview before committing:

```bash
$ video-cli edit-preview "add rain to the scene"
# Returns: before.jpg, after.jpg for comparison
# You decide if it looks right

$ video-cli edit "add rain to the scene"  # Apply for real
```

This is especially useful for:
- Edits that might break the scene
- Testing if the edit model understands the instruction
- Comparing multiple edit instructions before picking one

---

## Agent-Assisted Iteration

For automated prompt refinement, use VLM-in-the-loop:

```python
def refine_prompt(goal: str, initial_prompt: str, max_iterations: int = 10):
    """Iterate on prompt until visual goal is achieved."""

    current_prompt = initial_prompt
    run(f'video-cli prompt "{current_prompt}"')

    for i in range(max_iterations):
        # Generate
        run("video-cli step")

        # Evaluate via VLM
        result = json.loads(run("video-cli describe-frame"))
        description = result["description"]

        # Check if goal is met
        if goal_achieved(description, goal):
            print(f"Goal achieved at iteration {i}")
            return current_prompt

        # Analyze gap and adjust
        adjustment = analyze_gap(description, goal)
        current_prompt = adjust_prompt(current_prompt, adjustment)
        run(f'video-cli prompt "{current_prompt}"')

    print("Max iterations reached")
    return current_prompt
```

Example goals:
- "hand should be trembling" → check if description mentions trembling
- "prop should be visible in frame" → check if description mentions prop
- "character should look scared" → check for fear-related descriptors

---

## What You're Testing

| Test Type | What You Vary | What You Observe |
|-----------|---------------|------------------|
| **Prompt phrasing** | Word choice, structure | Does it produce the intended visual? |
| **Seed** | Random seed | Better/worse motion, composition |
| **LoRA vocabulary** | Trigger words, style descriptors | Does this LoRA respond to this phrasing? |
| **Strength/scale** | LoRA scale, guidance | How much influence? |
| **Transition** | Soft vs hard cut | How does it affect continuity? |

---

## Comparison Modes

### Side-by-Side

Generate branches, export key frames, compare visually:
```bash
$ video-cli fork --variations 4 --horizon 6
# Generates 4 variations, 6 chunks each
$ video-cli compare --out comparison.jpg
# Grid of key frames from each branch
```

### A/B Testing

Sequential comparison of two approaches:
```bash
$ video-cli snapshot  # Save baseline
$ video-cli prompt "approach A"
$ video-cli run --chunks 5
$ video-cli export approach_a.mp4

$ video-cli restore snap-001
$ video-cli prompt "approach B"
$ video-cli run --chunks 5
$ video-cli export approach_b.mp4

# Watch both, decide
```

### Metric-Based (Future)

If you have a way to score outputs:
```bash
$ video-cli evaluate --metric motion_amount
# Returns: { "motion_score": 0.73 }
```

Could integrate with VLM for semantic evaluation:
```bash
$ video-cli evaluate --goal "hand should be trembling"
# Returns: { "goal_match": 0.85, "reasoning": "visible micro-movements in hand region" }
```

---

## Documentation Pattern

When you find a prompt that works, document it:

```yaml
# prompts/behaviors/trembling_hand.yaml
behavior: "trembling hand"
lora: "rat_v1"
working_prompts:
  - text: "character grips cup with shaky hands, visible tremor"
    effectiveness: 0.9
    notes: "works best with seeds 1000-2000 range"
  - text: "nervous hands holding cup, subtle shake, micro-movements"
    effectiveness: 0.7
    notes: "more subtle, good for background characters"
failed_prompts:
  - text: "hand trembling slightly"
    notes: "too subtle, often ignored by model"
  - text: "shaking violently"
    notes: "too extreme, looks like seizure"
seed_recommendations: [1234, 5678, 9012]
```

Over time, build a library of known-working prompts per behavior per LoRA.

---

## Relationship to Other Concepts

| Concept | Focus | This Workflow's Role |
|---------|-------|---------------------|
| **Narrative Engine** | Story coherence | N/A - different use case |
| **WorldState** | What's happening | This tests *how to express* what's happening |
| **StyleManifest** | Per-LoRA vocabulary | This *discovers* what vocabulary works |
| **VLM Integration** | Frame analysis | Enables automated evaluation in this workflow |
| **Branching** | Fork/compare/select | Core mechanic of this workflow |

---

## Implementation Dependencies

Already have:
- [x] `video-cli step/run/pause` - basic control
- [x] `video-cli prompt` - set prompt
- [x] `video-cli snapshot/restore` - branching basics

Need:
- [ ] `video-cli fork --variations N` - generate multiple branches
- [ ] `video-cli compare` - visual comparison of branches
- [ ] `video-cli edit-preview` - preview edit without applying
- [ ] `video-cli describe-frame` - VLM evaluation (in `proposals/vlm-integration.md`)
- [ ] `video-cli seed` - explicit seed control

---

## Open Questions

1. **How to present comparisons?** Grid image? Side-by-side video? Web UI?

2. **Automated variation generation?** Given a prompt, auto-generate variations:
   - Synonym substitution
   - Reordering
   - Adding/removing emphasis words

3. **Prompt library format?** YAML? JSON? Markdown with frontmatter?

4. **Integration with Codex?** Could Codex help iterate on prompts based on evaluation?

---

## Related Files

| File | Relationship |
|------|--------------|
| `notes/concepts/narrative-engine.md` | Different use case (story vs visual behavior) |
| `notes/proposals/vlm-integration.md` | Enables `describe-frame` evaluation |
| `notes/proposals/frame-buffer-scrubbing.md` | Enables visual comparison |
| `notes/research/2025-12-26/dev_console/claude_01.md` | Source doc |
