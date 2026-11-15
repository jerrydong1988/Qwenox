# Qwenox launcher

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Embedded HTML, CSS and JavaScript served by a C++ Windows launcher. The existing
native supervisor owns engine/API processes. This record scopes the launcher;
the separate inference API console and model kernels have their own behavior.

## Users

Jerry maintains local Flash HGN, 27B HGN and Flash GGUF inference configurations
on Windows. He wants a practical launcher with model start/stop and real-time
performance on its first screen, with configuration on separate pages.

## Product Purpose

Select a saved model profile, start the right local service, see its actual
state and throughput, and stop it without leaving child processes behind.

## Operating Context

The maintained branch supports HGN and Flash qwen4exp GGUF. Saved profiles carry
model-specific numerical policies. The API provides live counters and persisted
request statistics. Profile edits affect the next start, not the active model.

## Capabilities and Constraints

- Preserve profile editing, validation, private flags, draft/save/revert,
  start/stop/restart, file browsing, language selection and logs.
- Show actual API measurements. Distinguish live rates, request progress and
  historical aggregates. Missing or stale telemetry is not a measured zero.
- Keep monitoring read-only; do not change GPU kernels or daily deployments
  as a side effect of a launcher redesign.
- Keep external processes outside this launcher's stop ownership.

## Brand Commitments

Qwenox remains the name. Jerry explicitly rejects anime characters and imagery
and requests a conventional, professional launcher. No decorative wallpaper.

## Evidence on Hand

`src/api/live_stats.h` and `/metrics/live` provide active-request counters;
`/reqstat/summary` and `/reqstat/tail` provide historical measurements. Existing
launcher tests and isolated model runtimes are available for validation.

## Product Principles

- Put launch controls and measured runtime state first.
- Keep configuration separate from observation.
- Preserve data, model policies and process ownership.
- Prefer familiar controls and clear labels over decoration.
