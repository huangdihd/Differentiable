# Functional editor shell

`App.tsx` connects the formal chapter 0/1/2 catalog and XOR preview to the Worker. Interactive tensor execution happens in the Worker; one-time legacy save migration validates old circuits asynchronously before the editor becomes available. `Canvas.tsx` only edits/displays JSON graphs. `Properties.tsx` handles node attributes; loss rendering reads `--accent` from the existing theme. `layout.css` contains layout rules only; `theme.css` is owned by the visual designer.

## Controls

- Drag a palette component onto the canvas (or click to add).
- Drag output to input, or onto a node's first empty input. Drag a connected input to rewire atomically; a rejected/cancelled connection preserves the original edge.
- Click a wire to remove it. Input/output interface nodes cannot be deleted.
- Click / Shift-click selects; Shift-drag blank space adds intersecting nodes to the selection. Clicking the background clears selection; middle-button dragging also pans. Drag blank space, or hold Space and drag, to pan. Wheel zoom anchors at the pointer.
- Delete, Ctrl/Cmd+C/V, Ctrl/Cmd+Z, Ctrl/Cmd+Shift+Z. Clipboard is local to the editor, not the system clipboard.
- Expressions insert a laid-out fragment below existing nodes and connect it to the level output. Double-click a subgraph component to inspect its implementation read-only; Back returns.
- Tests/probes and training run in a module Worker. Training edits are locked until stopped or completed; pause retains optimizer state.
- Live values keeps scalar inputs editable and displays tensor fixtures read-only with previous/next case controls. Tensor nodes show shape, a short preview, and complete values in a tooltip. Shape-variant failures name the tested input shapes in the results table.
- Size (`|x|ₖ`) starts at axis −1 and accepts integer axis edits. Reshape exposes optional dimension ports d0–d3 only up to its target rank; wired shape overrides and wired range lengths are marked with arrows and disable their attribute field (disconnect to edit). Optional inputs do not produce missing-wire notes.
- Sum and reduce-max start with axis −1 and Keep dimensions enabled. Unsqueeze starts at axis −1; its axis editor accepts negative integers (valid range is checked against the input rank during compilation).
- Loading an existing v2 save migrates chapter-2 IDs to the eight-lesson route and persists v3. Loading progress then refreshes unlocks from already-passed lessons, so new successors become available without replaying. New transpose nodes swap axes −2 and −1; legacy full permutations remain editable.

## State and persistence

`src/editor/state.ts` is a pure command reducer with graph/selection/viewport, snapshots for undo/redo, deterministic IDs, internal clipboard and cycle/port validation. A node drag previews locally and commits one move command. Pan/zoom is reducer state but is not graph history. Progress uses the engine's pure `applyResult`, including its best-score ordering. Only passing results unlock levels/components. The exact submitted graph is saved independently of unfinished edits in `passedSolutions`; passing again replaces that implementation. The palette groups allowed primitives, previously built components and level presets, excluding forbidden components and the current lesson's output. Par and optional feature-unlock messages appear on completion.

Graphs include positions in `node.ui`. `src/storage.ts` validates versioned localStorage data and catches unavailable/corrupt storage. A trailing throttle writes edits every 400ms, with a pagehide/unmount flush. A failed save does not crash the page; language persistence is attempted independently even when writing progress fails. Read-only component inspection also disables undo/redo on the parent graph. Probe requests discard results when the graph or player-registry revision changes. The editor and canvas share `buildPlayerRegistry`; registry changes are sent to the Worker with `syncProgress` before compilation. Component inspection displays the player's passed graph, including nested implementations.

## Localization

Use `const t = createTranslator(locale)` then `t('ui.report', {step, loss, accuracy})`. Numeric interpolation uses `Intl.NumberFormat`; plural dictionaries select via `Intl.PluralRules` and a `count` parameter. `translateDiagnostic(locale, error)` maps engine codes, without displaying exception prose.

Add `src/locales/<language>.json` with the same flat keys as English and register its import in `src/i18n.ts`. Locale names are derived from the locale registry. Missing messages fall back to English and warn in development. Tests check Chinese/English key parity.

## Validation boundaries and remaining UI polish

The environment cannot run a browser: builds and Node tests do not prove pointer interactions, accessibility or WebGPU behavior in an actual browser. Visual review is required. No motion has been added; rejected cycles show a translated error rather than a node shake because the supplied theme has no shake class. Large graphs do not have virtualization, minimap or fit-to-view controls. Inspecting shape-dependent generated subgraphs can fail without concrete input shape context; it reports an error instead of guessing shapes. Probe results currently show the first output port per node and a bounded numeric preview, not live training parameters. The formal scalar/derivative lessons use length-8 vector inputs; the XOR preview uses batches.
