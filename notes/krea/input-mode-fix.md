# Fix: Prompt input locked after switching input modes

## Problem
When switching from video to text input mode, the prompt input field remains locked with the default "panda" prompt. This happens because:
1. Timeline state persists across mode switches
2. The `paused` state persists
3. The disabled logic in `InputAndControlsPanel.tsx` locks the input when: `!selectedTimelinePrompt && isVideoPaused && !isAtEndOfTimeline()`

## Solution
In `StreamPage.tsx`, add the following to `handleInputModeChange` after the `stopStream()` call:

```tsx
// Reset timeline completely when switching modes to avoid stale state
if (timelineRef.current) {
  timelineRef.current.resetTimelineCompletely();
}

// Reset selected timeline prompt to exit Edit mode and return to Append mode
setSelectedTimelinePrompt(null);
setExternalSelectedPromptId(null);
```

And add `paused: false` to the `updateSettings()` call:

```tsx
updateSettings({
  inputMode: newMode,
  resolution,
  denoisingSteps: modeDefaults.denoisingSteps,
  noiseScale: modeDefaults.noiseScale,
  noiseController: modeDefaults.noiseController,
  paused: false,  // Add this line
});
```

Also update the comment above `updateSettings` to:
```tsx
// Update settings with new mode and ALL mode-specific defaults including resolution
// Also reset paused state so prompt input is not locked
```

## Files affected
- `frontend/src/pages/StreamPage.tsx`
