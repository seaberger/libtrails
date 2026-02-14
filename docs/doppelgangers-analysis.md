# Doppelgangers Visualization Analysis

Research notes on [badlogic/doppelgangers](https://github.com/badlogic/doppelgangers) for improving the LibTrails galaxy homepage.

**Live demos**: [openai/codex](https://mariozechner.at/uploads/codex.html) | [sst/opencode](https://mariozechner.at/uploads/opencode.html) | [openclaw/openclaw](https://mariozechner.at/uploads/openclaw.html)

---

## How It Works

The tool fetches GitHub issues/PRs, generates OpenAI embeddings, projects them via UMAP to 2D/3D, and outputs a self-contained HTML file with an interactive viewer. The entire codebase is 3 files (~1,700 lines total).

| File | Purpose | Lines |
|------|---------|-------|
| `src/triage.ts` | CLI orchestrator — fetches items via `gh` | 477 |
| `src/embed.ts` | Generates embeddings (OpenAI or local GGUF) | 224 |
| `src/build.ts` | UMAP projection + generates HTML viewer | 1020 |

The viewer is ~520 lines of vanilla JS embedded in the HTML template (`build.ts:158-981`). No React, no Three.js, no build step for the viewer — just Canvas 2D and hand-written event handlers.

---

## Rendering Approach

### Canvas 2D (not WebGL/Three.js)

The entire visualization uses a 2D canvas context (`ctx = canvas.getContext("2d")`). Points are drawn with simple `ctx.arc()` calls.

```js
// build.ts:716-729 — the entire point rendering loop
for (const item of projected) {
  if (item.screen.culled) continue;
  const point = data[item.index];
  const isSelected = state.selected.has(item.index);
  const size = isSelected ? 4 : 2.5;    // <-- tiny dots!
  ctx.fillStyle = getPointColor(point, isSelected);
  if (point.type === "issue") {
    ctx.strokeStyle = getPointColor(point, isSelected);
    drawRing(item.screen.x, item.screen.y, size + 1);  // hollow ring
  } else {
    ctx.beginPath();
    ctx.arc(item.screen.x, item.screen.y, size, 0, Math.PI * 2);
    ctx.fill();  // filled circle
  }
}
```

### Point Sizes
- **Normal**: 2.5px radius (very small — emphasizes cluster structure over individual items)
- **Selected**: 4px radius (subtle size increase, mainly distinguished by color change to orange)

### Visual Encoding
- **Green** (#6ee7b7) = Open items
- **Purple** (#a78bfa) = Closed items
- **Orange** (#f59e0b) = Selected items (overrides state color)
- **Filled circles** = PRs
- **Hollow rings** = Issues (stroke-only, 1.5px line width)

### 3D Mode
Hand-rolled rotation matrix projection (lines 580-600):
- Manual cosY/sinY/cosX/sinX rotation
- Perspective division: `1 / (1 + z * 0.9)`
- Points behind camera (z < -0.9) are culled
- Depth sorting for painter's algorithm occlusion
- No lighting, no materials — just colored dots with perspective scaling

### Performance
- `requestAnimationFrame` scheduling prevents redundant renders (lines 686-694)
- Only renders when state changes (drag, zoom, selection)
- Simple clear + redraw loop (no incremental updates)
- Works well for ~1,000-5,000 points

---

## Interaction Model

### Key Insight: NO Hover Tooltips

Doppelgangers has **zero hover interaction**. All information display happens through **click/selection → sidebar panel**. This is a major design difference from LibTrails' current hover-tooltip approach.

### Selection Mechanics (lines 746-848)

| Action | Effect |
|--------|--------|
| **Drag** | Pan (2D) or Rotate (3D) |
| **Ctrl/Cmd+Drag** | Pan (3D mode only) |
| **Shift+Drag** | Rectangle select — dashed blue outline appears |
| **Ctrl/Cmd+Shift+Drag** | Add to existing selection |
| **Click empty space** | Deselect all |
| **Scroll** | Zoom |

The rectangle selection (lines 653-671):
```js
const selectPoints = () => {
  if (!state.selectRect) return;
  const rect = state.selectRect;
  // ... compute bounds
  if (!state.addToSelection) state.selected.clear();
  for (let i = 0; i < data.length; i++) {
    if (!isVisible(data[i])) continue;
    const screen = getScreenPoint(data[i]);
    if (screen.x >= left && screen.x <= right &&
        screen.y >= top && screen.y <= bottom) {
      state.selected.add(i);
    }
  }
  updateSidebar();
};
```

### Click Disambiguation (lines 818-848)

The `endDrag` handler distinguishes actual clicks from drags using distance:
```js
const totalDist = Math.hypot(point.x - mouseDownPos.x, point.y - mouseDownPos.y);
if (totalDist < 3) {
  // This was a click, not a drag
  const hitIndex = hitTestPoint(point.x, point.y, 8);
  if (hitIndex === -1) {
    state.selected.clear();  // clicked empty space → deselect
    updateSidebar();
  }
}
```

Note: Single-click on a point does NOT select it in the current code. Selection is rectangle-only. (This seems like a missing feature — LibTrails could do better here.)

---

## Sidebar Panel (lines 289-366, 607-651)

Fixed 480px right panel, always visible:

```
┌─────────────────────────────────┐
│ Selection          [Open All] [Copy] │
│ 5 selected                          │
│                                     │
│ ┌─────────────────────────────┐     │
│ │ #1234 Fix auth bug  [PR][Open]│   │
│ │ Fixed the token refresh...   │    │
│ └─────────────────────────────┘     │
│ ┌─────────────────────────────┐     │
│ │ #1235 Auth timeout  [Issue][Open]│ │
│ │ Users report timeout when...  │   │
│ └─────────────────────────────┘     │
│ ...                                 │
└─────────────────────────────────┘
```

Each selected item shows:
- Linked title (opens in new tab)
- Type badge (PR / Issue) with color coding
- State badge (Open / Closed)
- Body snippet (3 lines max, CSS `-webkit-line-clamp: 3`)

Action buttons:
- **Open All**: Opens every selected item's URL in new tabs
- **Copy**: Copies formatted list to clipboard

---

## Filtering (lines 428-432, 509-519)

Top-left HUD with checkbox toggles:
- `[ ] PRs` / `[ ] Issues` — toggle type visibility
- `[ ] Open` / `[ ] Closed` — toggle state visibility

Visibility is checked before rendering each point:
```js
const isVisible = (point) => {
  const typeOk = !hasTypes ||
    (point.type === "pr" && filterPr.checked) ||
    (point.type === "issue" && filterIssue.checked);
  const stateOk = !hasStates ||
    (point.state === "open" && filterOpen.checked) ||
    (point.state === "closed" && filterClosed.checked);
  return typeOk && stateOk;
};
```

---

## Semantic Search (lines 882-947)

If `--search` flag was used (includes embeddings in HTML), a search box appears:
1. User types a query
2. Browser calls OpenAI embedding API directly (asks for API key on first use)
3. Computes cosine similarity against all point embeddings client-side
4. Selects top 20 matches → highlighted + shown in sidebar

---

## Recommendations for LibTrails

### High Priority — Should Implement

#### 1. Click-to-Select (Replace Hover Tooltip)
**Current**: Hover over a sphere → ephemeral tooltip appears, disappears when you move away.
**Proposed**: Click a sphere → it stays selected (highlighted), info persists in a side panel. Click empty space to deselect. This is strictly better UX — the tooltip is hard to read while the mouse is moving.

**Implementation**: In `GalaxyView.tsx`, change `onPointerMove` handler to `onClick`. Track `selectedCluster` state instead of `tooltip` state. Render a persistent panel instead of a floating div.

#### 2. Selection Sidebar Panel
**Current**: Floating tooltip with limited info.
**Proposed**: Right-side panel (300-400px) showing:
- Cluster name + domain badge
- Book count + topic count
- Top 5 topics as chips
- Featured book covers (clickable → book detail page)
- Link to full cluster page

The domain legend could move INTO this panel (below the selection info, or in a collapsible section), reclaiming canvas space.

#### 3. Smaller Spheres
**Current**: Spheres range from `0.3 + 1.2 * sqrt(book_count / maxBooks)` — the large ones are very prominent.
**Proposed**: Reduce the size range significantly. Doppelgangers uses 2.5px dots. For our 3D scene, something like `0.15 + 0.6 * sqrt(ratio)` would show cluster structure better without individual spheres dominating the view.

#### 4. Multi-Select with Shift+Drag
**Proposed**: Shift+drag draws a rectangle overlay on the canvas. All clusters within the rectangle get selected and listed in the sidebar. Ctrl+Shift adds to selection. This lets users explore neighborhoods: "What clusters are in this corner of the galaxy?"

### Medium Priority — Worth Considering

#### 5. 2D/3D Toggle
Doppelgangers supports both modes with one button. Our UMAP already generates 3D coords — we could add a "flatten to 2D" mode that just uses x,y with an orthographic camera. Some users may find 2D easier to navigate.

#### 6. Semantic Search Highlighting
Add a search box to the galaxy view. User types a query → API finds matching clusters by semantic similarity → those clusters glow/pulse/enlarge. This connects the search functionality we already have to the visualization.

#### 7. Filter Checkboxes for Domains
Replace the current clickable domain legend with checkbox toggles (or keep it but make the UX clearer). The doppelgangers approach of simple checkboxes is very clean and obvious.

### Lower Priority — Nice to Have

#### 8. Selection Actions
- "View All" button: Opens all selected clusters in new tabs
- "Copy" button: Copies cluster names/URLs to clipboard
- These are trivial to add once the selection model exists

#### 9. requestAnimationFrame Scheduling
Doppelgangers only re-renders when state changes. Our Three.js OrbitControls currently runs an animation loop. Adding `autoRotate` already does this, but we could optimize by only rendering on interaction.

### Not Applicable

- **Canvas 2D rendering**: Our Three.js approach is better for 3D with lighting/materials
- **Self-contained HTML**: Our architecture (Astro + React) is more maintainable
- **PR/Issue type encoding**: Different domain, no equivalent in LibTrails

---

## Key Design Takeaway

Doppelgangers prioritizes **information density and utility** over visual spectacle. Small dots reveal cluster structure. Persistent selection replaces ephemeral hover. A scrollable sidebar replaces a tiny tooltip. These choices make it a genuinely useful triage tool rather than just a pretty visualization.

LibTrails should adopt this philosophy: the galaxy view should be a **tool for exploring your library**, not just eye candy. Click-to-select + sidebar panel + multi-select would transform it from a screensaver into an interactive discovery interface.
