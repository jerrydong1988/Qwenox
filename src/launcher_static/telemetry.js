/* Shared counter semantics for the launcher and its regression test. */
(function(root){
  const numeric = x => typeof x === "number" && Number.isFinite(x) && x >= 0;
  function observe(previous, sample){
    if (!sample || typeof sample.session !== "string" || !numeric(sample.sample_ms) ||
        !Array.isArray(sample.active) ||
        !["prefill_estimated_tokens_total", "output_tokens_total"].every(k => numeric(sample[k])))
      throw new Error("Invalid live telemetry");
    const dt = previous ? (sample.sample_ms - previous.sample_ms) / 1000 : 0;
    const continuous = previous && previous.session === sample.session && dt > 0 && dt < 5;
    const rate = k => continuous && sample[k] >= previous[k] ? (sample[k] - previous[k]) / dt : null;
    return {sample, reset:!continuous, decode:rate("output_tokens_total"),
      prefill:sample.active.some(r => !r.prefill_reported) ? null : rate("prefill_estimated_tokens_total"),
      requests:sample.active.length, output:sample.output_tokens_total};
  }
  function acceptance(summary){
    let accepted=0, proposed=0;
    for (const d of Object.values(summary?.drafters || {})) {
      if (numeric(d.accepted) && numeric(d.proposed) && d.accepted <= d.proposed) {
        accepted += d.accepted; proposed += d.proposed;
      }
    }
    return proposed ? accepted/proposed : null;
  }
  const api = {observe, acceptance, numeric};
  if (typeof module === "object" && module.exports) module.exports=api;
  else root.QwenoxTelemetry=api;
})(typeof globalThis === "object" ? globalThis : this);
