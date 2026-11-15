---
name: "Qwenox Launcher"
description: "A familiar Windows launcher for local model operation and measured runtime state."
colors:
  ink: "#202936"
  dim: "#596678"
  faint: "#647184"
  line: "#e2e6ec"
  line2: "#bac3cf"
  card: "#fff"
  card2: "#f5f7fa"
  inp: "#fff"
  hover: "#edf1f6"
  sel: "#e8f0fc"
  accent: "#185fc1"
  btn: "#1b61bf"
  btn-hi: "#174e99"
  ok: "#187744"
  bad: "#bc303a"
  warn: "#926014"
typography:
  headline:
    fontFamily: "Segoe UI, Microsoft YaHei, system-ui, sans-serif"
    fontSize: "22px"
    fontWeight: 600
    lineHeight: 1.3
    letterSpacing: "-0.02em"
  title:
    fontFamily: "Segoe UI, Microsoft YaHei, system-ui, sans-serif"
    fontSize: "14px"
    fontWeight: 600
    lineHeight: 1.5
  body:
    fontFamily: "Segoe UI, Microsoft YaHei, system-ui, sans-serif"
    fontSize: "14px"
    fontWeight: 400
    lineHeight: 1.5
  label:
    fontFamily: "Segoe UI, Microsoft YaHei, system-ui, sans-serif"
    fontSize: "12px"
    fontWeight: 400
    lineHeight: 1.5
  control:
    fontFamily: "Segoe UI, Microsoft YaHei, system-ui, sans-serif"
    fontSize: "13px"
    fontWeight: 400
    lineHeight: 1.4
  measurement:
    fontFamily: "Segoe UI, Microsoft YaHei, system-ui, sans-serif"
    fontSize: "26px"
    fontWeight: 600
    lineHeight: 1.3
    letterSpacing: "-0.02em"
  log:
    fontFamily: "Consolas, Cascadia Mono, monospace"
    fontSize: "12px"
    fontWeight: 400
    lineHeight: 1.6
rounded:
  tag: "3px"
  field: "4px"
  control: "5px"
  profile: "6px"
  dialog: "10px"
spacing:
  "4": "4px"
  "8": "8px"
  "12": "12px"
  "16": "16px"
  "20": "20px"
  "24": "24px"
  "28": "28px"
components:
  button-primary:
    backgroundColor: "{colors.btn}"
    textColor: "{colors.card}"
    typography: "{typography.control}"
    rounded: "{rounded.control}"
    padding: "7px 14px"
  button-primary-hover:
    backgroundColor: "{colors.btn-hi}"
  button-secondary:
    backgroundColor: "{colors.card}"
    textColor: "{colors.ink}"
    typography: "{typography.control}"
    rounded: "{rounded.control}"
    padding: "7px 14px"
  button-secondary-hover:
    backgroundColor: "{colors.hover}"
  button-danger:
    backgroundColor: "{colors.card}"
    textColor: "{colors.bad}"
    typography: "{typography.control}"
    rounded: "{rounded.control}"
    padding: "7px 14px"
  text-input:
    backgroundColor: "{colors.inp}"
    textColor: "{colors.ink}"
    typography: "{typography.control}"
    rounded: "{rounded.field}"
    padding: "7px 9px"
  tab-selected:
    textColor: "{colors.accent}"
    padding: "13px 12px 11px"
  profile-selected:
    backgroundColor: "{colors.sel}"
    textColor: "{colors.ink}"
    rounded: "{rounded.profile}"
    padding: "12px 12px"
  chip-selected:
    backgroundColor: "{colors.sel}"
    textColor: "{colors.accent}"
    typography: "{typography.label}"
    rounded: "{rounded.field}"
    padding: "4px 8px"
  measurement:
    textColor: "{colors.ink}"
    typography: "{typography.measurement}"
  status-running:
    textColor: "{colors.ok}"
  dialog:
    backgroundColor: "{colors.card}"
    textColor: "{colors.ink}"
    rounded: "{rounded.dialog}"
    padding: "24px"
    width: "min(460px,92vw)"
---

# Design System: Qwenox Launcher

## Overview

**Creative North Star: "The Practical Desktop Launcher"**

Qwenox is a familiar Windows utility: a white workspace, a cool grey profile rail, readable dark labels and restrained blue controls. Compact Segoe UI typography supports repeated operational use. Visual emphasis comes from state, selection and measured values rather than decoration.

This system applies only to the embedded launcher in src/launcher_static. It does not establish the design of the separate inference API dashboard. The user has confirmed a professional launcher with no anime imagery or decorative wallpaper; the implementation is code-led and ships no ornamental raster assets.

**Key Characteristics:**

- Light working surfaces with a cool profile rail.
- Blue actions and selection; semantic colours for service state.
- Compact Windows typography with tabular numerical measurements.
- Flat workspace panels, with elevation reserved for transient feedback.
- Clear separation between a selected profile and the running service.

The normative tokens above are extracted from [launcher.css](src/launcher_static/launcher.css). Component behaviour comes from [index.html](src/launcher_static/index.html) and [telemetry.js](src/launcher_static/telemetry.js). [PRODUCT.md](PRODUCT.md) and the [launcher surface brief](.impeccable/surfaces/src-launcher-static-index-html.md) supply the confirmed scope and direction. The sidecar's generated tonal ramps are inspection aids, not additional implementation tokens.

## Colors

The palette combines cool white and grey working surfaces with a clear blue action family. Semantic colours carry operational meaning rather than decorative variation.

### Primary

- **Action Blue** (`btn`): filled primary buttons and the throughput line. **Pressed Blue** (`btn-hi`) is the button hover colour.
- **Selection Blue** (`accent`): links, active tab text and underline, progress, and focus indication.
- **Pale Selection** (`sel`): selected profiles, chips and options. It supports blue text without competing with primary actions.

### Neutral

- **Workspace White** (`card`) and **Field White** (`inp`): main surfaces and editable fields. These distinct source roles intentionally share a value.
- **Cool Rail** (`card2`): profile navigation and log ground.
- **Hover Grey** (`hover`): secondary-control and row hover treatment.
- **Ink** (`ink`), **Secondary Ink** (`dim`) and **Quiet Ink** (`faint`): main content, supporting labels and tertiary metadata.
- **Fine Divider** (`line`) and **Control Outline** (`line2`): structural separation and input/secondary-control boundaries.

### Semantic states

- **Running Green** (`ok`): running/healthy labels and indicators.
- **Attention Amber** (`warn`): busy states, unsaved changes, next-start notices and telemetry warnings.
- **Error Red** (`bad`): validation failures, destructive controls and failed results.

**The Action and State Rule.** Use blue for interaction and selection. Reserve green, amber and red for operational state, validation and feedback; pair them with text.

## Typography

**Body Font:** Segoe UI, with Microsoft YaHei, system UI and sans-serif fallbacks.
**Label Font:** The same UI stack; there is no separate display typeface.
**Mono Font:** Consolas, with Cascadia Mono and monospace fallbacks, for logs and technical keys.

**Character:** Familiar application typography at a compact working density. Headings identify the selected or running model; the largest text belongs to operational measurements rather than promotional display copy.

### Hierarchy

- **Headline** (`headline`): model heading. It reduces to (20px) on mobile.
- **Title** (`title`): settings sections and operational subsections. The product mark uses a separate modest heading (18px), and dialog titles use (17px).
- **Body** (`body`): base application text with a line-height of (1.5).
- **Control** (`control`): buttons and text fields. Profile names and most settings labels are (13px), semibold when they identify a field or model.
- **Label** (`label`): supporting labels, status and table text. Minor hints and units may use (11px); these are metadata, not primary task instructions.
- **Measurement** (`measurement`): live values with tabular numerals. Values reduce to (23px) at the compact breakpoint and become (25px) in the mobile two-column grid.
- **Log** (`log`): readable monospaced logs. Technical keys use a smaller contextual style and remain secondary to human-readable labels.

**The Measured Value Rule.** Use tabular numerals for measurements and numerical tables. Show an em dash when a value is unavailable; only measured zero is displayed as zero.

## Layout

The desktop shell fills the viewport. A top action bar has a minimum height of (64px); the profile rail has a fixed basis of (240px). The flexible workspace contains a horizontal tab strip, an independently scrolling content pane and a save bar when relevant. The profile list scrolls independently. Persistent divisions use borders rather than nested elevated cards.

The first launcher view places the model heading above four measurement columns, a throughput/request area and recent-request statistics. This is the launcher surface arrangement, not a required composition for the separate API dashboard. Parameters remain on Model, Inference and Machine tabs; logs remain separate.

At widths up to (1100px), the profile rail becomes (210px), content padding becomes (22px), settings labels narrow from (190px) to (155px), and secondary header context is hidden. The activity area uses equal columns instead of the wider plot proportion (1.2fr to 1fr).

At widths up to (760px), the document scrolls naturally. The profile rail becomes a horizontal list of (202px) profile items; the tab strip remains horizontally scrollable. Measurements become two columns, activity and forms become one column, content uses (22px 16px) padding, and the save bar sticks to the bottom when present. Tables retain horizontal scrolling rather than crushing numerical columns. The header wraps and keeps the service state on a dedicated row.

Spacing is compact and task-based. The frontmatter records commonly reused steps; this is not an enforced mathematical grid. Desktop content uses (24px 28px 28px) padding, settings rows use (12px) vertical padding, and related controls commonly use (8px) gaps. Form help stays next to its field. The plotting area and request area share a minimum height of (170px) on desktop.

## Elevation & Depth

The persistent workspace is flat. White and cool grey surfaces, fine dividers and selection fills establish hierarchy. Modality is the exception: dialogs use a translucent dark overlay and a soft shadow; toasts use a smaller shadow to stay distinct from the page beneath them. There are no decorative gradients, texture layers or background illustrations.

### Shadow Vocabulary

- **Dialog:** (`0 18px 60px #1420373d`) for both standard dialogs and the file browser.
- **Toast:** (`0 5px 20px #1a273520`) for transient success and error feedback.
- **Modal overlay:** (`#101b2f5e`) on the full-window native dialog surface; its browser backdrop is transparent to avoid double dimming.

**The Flat Workspace Rule.** Use borders and tonal separation for persistent surfaces. Reserve shadows for dialogs and toasts.

## Shapes

Small, softened rectangles are the default. The frontmatter distinguishes compact tags, fields, standard controls, profile items and dialog containers. Inputs and chips share the field radius; plots, empty states and logs share the standard control radius. Borders are usually (1px). The active tab uses a (2px) bottom line.

Circular geometry is functional: service-state dots are (7px) and the switch thumb is (14px) inside a rounded (32px by 18px) track. These are not ornamental motifs. Dialog cards are visibly softer than everyday controls; the wider file browser keeps the same dialog radius.

## Components

### Buttons

Familiar, compact and explicit. Primary buttons use Action Blue with white text; secondary buttons use white with a stronger neutral outline; destructive buttons use red text and a pale red outline on white. Standard buttons have a minimum height of (33px), while compact utility buttons use (28px). Primary launch controls use semibold text and a minimum width of (100px).

Hover darkens the primary fill or adds a light ground to secondary/destructive controls. Active buttons move down by (1px). Disabled buttons use reduced opacity (0.45), a disabled cursor and no active transform. Keyboard focus uses a (2px) blue outline offset by (3px).

### Chips and segmented options

Compact, lightly outlined controls. Selected chips use the pale selection ground and blue text; hover strengthens the border. Segmented options share a single outlined enclosure with internal separators. Selection is indicated through fill and text weight rather than animation.

### Profile items and containers

Profile items combine a model name, state tag and short configuration summary. A selected item uses pale blue fill and a blue-grey boundary; an unsaved item uses a dashed boundary. Profile actions are revealed for the selected item or on keyboard focus within it. Names truncate in the rail; the main model heading may wrap.

Workspace sections use headings and dividers. Plot and empty-state grounds are subtly tinted. A profile item is not an invitation to give every section a card border or shadow.

### Inputs / Fields

White, clearly outlined fields with human-readable labels, adjacent help and visible validation. Text fields use compact padding and the field radius. Number fields use tabular numerals and a desktop basis of (140px). Selects retain the browser's familiar affordance.

Focus combines a blue boundary with a pale blue outline (`2px solid #cadcf7`, offset `1px`). Invalid fields use a red boundary plus an error message; edited rows use a light warm tint and expose a reset control. Disabled fields use a muted ground and tertiary text. Switches use a short thumb transition (`left .15s`); reduced-motion preferences remove transitions and animations.

### Navigation

Profile navigation is spatially separate from the workspace tabs. Tabs use dark secondary labels, a faint hover ground and blue text with an underline when selected. Text stays unwrapped and scrolls horizontally when needed. The current profile context at the far edge disappears at the compact breakpoint.

### Runtime measurements and history

Four large values pair labels, units and explanatory hints. The throughput plot uses a blue line, light horizontal guides, an accessible text label and explicit time/unit captions. It appears after sufficient valid samples and breaks the line at missing samples. Request progress uses native progress controls with accessible labels.

Live telemetry, estimated prefill and historical aggregates have distinct captions. Missing data uses an em dash and an explanatory empty state, while a valid idle sample may report zero. Historical totals are identified as belonging to the runtime directory; cached results carry a read time and last-result wording when appropriate. Tables align numerical columns right and the request/result column left.

The running model and endpoint remain tied to the active service when the user selects a different saved profile. The pending profile is shown as the next start, using attention text.

### Dialogs and feedback

Standard dialogs use a card width of `min(460px,92vw)` and the file browser uses `min(760px,94vw)` with a maximum height of (82vh). Actions align right and wrap when necessary. The implementation uses named native `dialog` elements opened with `showModal()`, initial control focus, Tab containment, Escape dismissal and focus restoration to the invoking control or active tab.

Success and error toasts combine tinted grounds, explicit text and matching boundaries. Validation messages remain next to their inputs; colour is never the only explanation.

## Do's and Don'ts

### Do:

- **Do** use the source colour roles and preserve their readable contrast on light surfaces.
- **Do** keep launch actions and runtime observations easy to find, with configuration on its own tabs.
- **Do** label live measurements, estimated prefill and historical statistics distinctly.
- **Do** keep the running model and endpoint visible when a different profile is selected for the next start.
- **Do** preserve visible focus, native dialog modality, Escape dismissal and focus restoration.
- **Do** let the profile rail, measurement grid and form rows reflow at the implemented breakpoints.

### Don't:

- **Don't** add anime characters, decorative wallpaper or ornamental raster imagery to the launcher.
- **Don't** substitute zero for unavailable telemetry or draw continuous throughput across missing samples.
- **Don't** turn every workspace section into a raised card.
- **Don't** make the separate API dashboard inherit this launcher-scoped document without a separate design decision.
