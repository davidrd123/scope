# Video CLI Cheat Sheet

## Quick Start

```bash
# Load a playlist and enter nav mode
video-cli playlist load prompts.txt
video-cli playlist nav

# Load with style swap (find/replace in all prompts)
video-cli playlist load prompts.txt --swap "Original Style" "New Style"
```

## Playlist Commands

```bash
video-cli playlist load <file>                      # Load caption file
video-cli playlist load <file> --swap "A" "B"       # Load with text swap
video-cli playlist status         # Show current state
video-cli playlist preview        # Show prompts around current
video-cli playlist next           # Next prompt
video-cli playlist prev           # Previous prompt
video-cli playlist goto <index>   # Jump to index
video-cli playlist apply          # Re-apply current prompt
video-cli playlist nav            # Interactive mode
```

## Nav Mode Keys

### Navigation
| Key | Action |
|-----|--------|
| `→` `n` `l` `SPACE` | Next |
| `←` `p` | Previous |
| `g` | Goto index |
| `a` | Apply current |
| `r` | Refresh |
| `q` `ESC` | Quit |

### Autoplay
| Key | Action |
|-----|--------|
| `o` | Toggle autoplay |
| `+` | Faster (min 1s) |
| `-` | Slower (max 30s) |

### Cut Modes
| Key | Mode | Effect |
|-----|------|--------|
| `h` | Hard cut | Reset cache (clean break) |
| `s` | Soft cut | Temp lower bias (faster adapt) |
| `t` | Transition | Embedding interpolation (morph) |
| `x` | One-shot | Single hard cut, no mode change |

### Soft Cut Tuning (`s` active)
| Bias | Key | | Chunks | Key |
|------|-----|-|--------|-----|
| 0.05 | `1` | | 1 | `!` |
| 0.10 | `2` | | 2 | `@` |
| 0.15 | `3` | | 3 | `#` |
| 0.20 | `4` | | 4 | `$` |
| 0.25 | `5` | | 5 | `%` |

### Transition Tuning (`t` active)
| Chunks | Key | | Method | Key |
|--------|-----|-|--------|-----|
| 1 | `6` | | Toggle LERP/SLERP | `T` |
| 2 | `7` | |
| 3 | `8` | |
| 4 | `9` | |
| 5 | `0` | |

## Other Commands

```bash
video-cli state                   # Session state
video-cli prompt "new prompt"     # Set prompt directly
video-cli prompt                  # Get current prompt
video-cli run                     # Start generation
video-cli pause                   # Pause generation
video-cli step                    # Generate one chunk
video-cli frame                   # Get current frame
video-cli style list              # List available styles
video-cli style set <name>        # Set active style
```

## Combo Recommendations

| Goal | Settings |
|------|----------|
| Quick responsive | `s` + bias=0.10, chunks=2 |
| Smooth morph | `s` + `t` + chunks=4 each |
| Clean scene break | `h` or `x` |
| Cinematic fade | `t` + SLERP + chunks=5 |

## LoRA Style Triggers

| LoRA | Trigger Phrase |
|------|----------------|
| Akira | `1988 Cel Animation` |
| Graffito | `Graffito Mixed-Media Stop-Motion` |
| Hidari | `Hidari Stop-Motion Animation` |
| Kaiju | `Japanese Kaiju Film` |
| Rooster & Terry | `Clay-Plastic Pose-to-Pose Animation` |
| TMNT | `Graffiti Sketchbook Animation` |

### Style Swap Examples

```bash
# Akira → Kaiju
video-cli playlist load akira_captions.txt \
  --swap "1988 Cel Animation" "Japanese Kaiju Film"

# Akira → Rooster & Terry
video-cli playlist load akira_captions.txt \
  --swap "1988 Cel Animation" "Clay-Plastic Pose-to-Pose Animation"

# TMNT → Akira
video-cli playlist load tmnt_captions.txt \
  --swap "Graffiti Sketchbook Animation" "1988 Cel Animation"
```
