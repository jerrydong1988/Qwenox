---
version: 1
slug: "src-launcher-static-index-html"
primary_target: "src/launcher_static/index.html"
related_targets: ["src/launcher_settings.cpp"]
---

# Launcher overview

Mode: Operate. Scope: src/launcher_static, read-only launcher telemetry bridge.
Audience: Jerry running local inference on Windows. Confirmed priority: model
start/stop plus live performance first; model parameters on separate pages.

## Direction contract

THESIS: A practical desktop launcher whose home explains what is running and
how it performs. Reject the wallpaper-first configuration screen.
OWN-WORLD: White workspace, cool light-grey profile rail, dark readable labels,
blue selection/actions, compact Segoe UI typography and tabular measurements.
STORY: Choose a model, launch it, observe live requests, inspect completed-request
statistics, then open configuration only when a setting needs adjustment.
FIRST VIEWPORT: 64px action bar; 240px profile rail; Overview tab first; selected
or active model heading, four measurement columns, a labelled throughput time
plot and active requests, followed by a compact recent-request table. Saved
settings stay on Model, Inference and Machine pages; logs have their own tab.
FORM: Familiar launcher convention explicitly chosen by the user's request and
confirmed task priority, overriding the random assignment. seed:9bd281660972dd3d.
Signature interaction: switching a saved profile changes the next-start selection
while the active model's counters and endpoint remain visibly pinned.
FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance

Unresolved: no user-specified colour mode; this session uses a light Windows
utility surface for readable settings and measurement tables. No ornamental
rasters ship. Live and historical metrics are labelled separately.
