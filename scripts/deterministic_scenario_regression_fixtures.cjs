function normalizeSuiteKey(value) {
  return String(value || "").trim().toUpperCase();
}

function normalizeScenarioRegressionFixture(input) {
  const suiteKey = normalizeSuiteKey(input.suiteKey);
  return {
    fixtureId: String(input.fixtureId || `${suiteKey.toLowerCase()}-${String(input.kind || "scenario")}`),
    kind: String(input.kind || ""),
    materialization: {
      count: Number(input.materialization?.count || 1),
      seed: Number(input.materialization?.seed || 1)
    },
    reason: String(input.reason || ""),
    replayKey: input.replayKey ? String(input.replayKey) : null,
    severity: input.severity ? String(input.severity) : null,
    suiteKey
  };
}

function formatRegressionFixtureEntry(entry) {
  const normalized = normalizeScenarioRegressionFixture(entry);
  const lines = [
    "{",
    `  fixtureId: "${normalized.fixtureId}",`,
    `  kind: "${normalized.kind}",`,
    `  materialization: { count: ${normalized.materialization.count}, seed: ${normalized.materialization.seed} },`,
    `  reason: "${normalized.reason.replace(/\\/g, "\\\\").replace(/"/g, "\\\"")}",`,
    `  suiteKey: "${normalized.suiteKey}"`
  ];
  if (normalized.replayKey) {
    lines.splice(lines.length - 1, 0, `  replayKey: "${normalized.replayKey}",`);
  }
  if (normalized.severity) {
    lines.splice(lines.length - 1, 0, `  severity: "${normalized.severity}",`);
  }
  lines.push("}");
  return lines.join("\n");
}

module.exports = {
  formatRegressionFixtureEntry,
  normalizeScenarioRegressionFixture,
  normalizeSuiteKey
};