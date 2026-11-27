const KIND_RANK = Object.freeze({
  internal: 0,
  change: 1,
  performance: 2,
  fix: 3,
  security: 3,
  feature: 4,
  breaking: 5
});

const DECISION_BY_KIND = Object.freeze({
  internal: "none",
  change: "patch",
  performance: "patch",
  fix: "patch",
  security: "patch",
  feature: "minor",
  breaking: "major"
});

const DECISION_RANK = Object.freeze({ none: 0, patch: 1, minor: 2, major: 3 });

const KIND_BY_CONVENTIONAL_TYPE = Object.freeze({
  bugfix: "fix",
  build: "internal",
  chore: "internal",
  ci: "internal",
  docs: "internal",
  enhancement: "feature",
  feat: "feature",
  feature: "feature",
  fix: "fix",
  hotfix: "fix",
  perf: "performance",
  performance: "performance",
  refactor: "internal",
  revert: "change",
  style: "internal",
  test: "internal"
});

const KIND_BY_CHANGELOG_SECTION = Object.freeze({
  added: "feature",
  changed: "change",
  deprecated: "feature",
  fixed: "fix",
  removed: "breaking",
  security: "security"
});

const HARD_SIGNAL_BY_SOURCE_KIND = Object.freeze({
  "title:breaking": "conventional-breaking-change",
  "body:breaking": "conventional-breaking-change",
  "commit:breaking": "conventional-breaking-change",
  "changelog:breaking": "changelog-removed-entry",
  "title:feature": "conventional-feature-commit",
  "commit:feature": "conventional-feature-commit",
  "changelog:feature": "changelog-added-entry"
});

const KIND_LABELS = Object.freeze({
  breaking: "breaking-change",
  feature: "enhancement",
  fix: "bug",
  performance: "performance",
  security: "security"
});

const DECLARED_SOURCES = new Set(["title", "body", "commit"]);
const RECORDED_SOURCES = new Set(["changelog"]);

const GITHUB_REVERT_TITLE_PATTERN = /^\s*revert\s+"/;
const CONVENTIONAL_HEADER_PATTERN = /^\s*([a-z][a-z0-9-]*)(\([^)]*\))?(!)?:\s*\S/;
const BREAKING_FOOTER_PATTERN = /^\s*breaking[\s-]change\s*:/im;
const BRANCH_TYPE_PATTERN = /^([a-z][a-z0-9-]*)\//;
const CHANGELOG_PATH_PATTERN = /(^|\/)(changelog|changes|history|news)(\.[^/]+)?$/;
const CHANGELOG_ENTRY_PATTERN = /^\s*[-*]\s+\S/;
const CHANGELOG_SECTION_PATTERN = /^###\s+(.+?)\s*$/;
const CHANGELOG_RELEASE_PATTERN = /^##\s+(?!#)/;
const HUNK_HEADER_PATTERN = /^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@(.*)$/;
const HTML_COMMENT_PATTERN = /<!--[\s\S]*?-->/g;

function withoutHtmlComments(text) {
  return String(text || "").replace(HTML_COMMENT_PATTERN, "");
}

function conventionalVotes(source, text, detail) {
  const header = String(text || "").split("\n")[0].toLowerCase();
  if (GITHUB_REVERT_TITLE_PATTERN.test(header)) {
    return [{ source, kind: KIND_BY_CONVENTIONAL_TYPE.revert, detail: `${detail} reverts an earlier change` }];
  }
  const match = header.match(CONVENTIONAL_HEADER_PATTERN);
  const votes = [];
  if (match && KIND_BY_CONVENTIONAL_TYPE[match[1]]) {
    votes.push({ source, kind: KIND_BY_CONVENTIONAL_TYPE[match[1]], detail: `${detail} is typed ${match[1]}` });
  }
  if (match && match[3]) {
    votes.push({ source, kind: "breaking", detail: `${detail} is marked breaking with "!"` });
  }
  return votes;
}

function footerVotes(source, text, detail) {
  return BREAKING_FOOTER_PATTERN.test(withoutHtmlComments(text))
    ? [{ source, kind: "breaking", detail: `${detail} has a BREAKING CHANGE footer` }]
    : [];
}

function changelogSectionOf(heading) {
  return KIND_BY_CHANGELOG_SECTION[String(heading || "").toLowerCase().replace(/[^a-z]/g, "")] || "";
}

function sectionsFromContent(content) {
  const sections = [];
  let current = "";
  String(content || "").split("\n").forEach((line, index) => {
    const section = line.match(CHANGELOG_SECTION_PATTERN);
    if (section) {
      current = section[1];
    } else if (CHANGELOG_RELEASE_PATTERN.test(line)) {
      current = "";
    }
    sections[index + 1] = current;
  });
  return sections;
}

function addedChangelogEntries(file) {
  const sections = file.headContent === undefined ? null : sectionsFromContent(file.headContent);
  const entries = [];
  let newLine = 0;
  let current = "";
  for (const line of String(file.patch || "").split("\n")) {
    const hunk = line.match(HUNK_HEADER_PATTERN);
    if (hunk) {
      newLine = Number(hunk[1]);
      const headerSection = hunk[2].trim().match(CHANGELOG_SECTION_PATTERN);
      current = headerSection ? headerSection[1] : "";
      continue;
    }
    if (line.startsWith("-") || line.startsWith("\\")) {
      continue;
    }
    const text = line.slice(1);
    const section = text.match(CHANGELOG_SECTION_PATTERN);
    if (section) {
      current = section[1];
    } else if (CHANGELOG_RELEASE_PATTERN.test(text)) {
      current = "";
    }
    if (line.startsWith("+") && CHANGELOG_ENTRY_PATTERN.test(text)) {
      entries.push(sections && sections[newLine] !== undefined ? sections[newLine] : current);
    }
    newLine += 1;
  }
  return entries;
}

function changelogVotes(files) {
  return files
    .filter((file) => CHANGELOG_PATH_PATTERN.test(String(file.filename || "").toLowerCase()) && file.status !== "removed")
    .flatMap((file) => {
      const counts = new Map();
      for (const heading of addedChangelogEntries(file)) {
        counts.set(heading, (counts.get(heading) || 0) + 1);
      }
      return [...counts].map(([heading, count]) => {
        const entries = count === 1 ? "a new entry" : `${count} new entries`;
        return {
          source: "changelog",
          kind: changelogSectionOf(heading) || "change",
          detail: heading
            ? `${file.filename} gains ${entries} under "${heading}"`
            : `${file.filename} gains ${entries}`
        };
      });
    });
}

function branchVotes(headRef) {
  const type = String(headRef || "").toLowerCase().match(BRANCH_TYPE_PATTERN)?.[1] || "";
  const kind = KIND_BY_CONVENTIONAL_TYPE[type];
  return kind ? [{ source: "branch", kind, detail: `the branch "${headRef}" is typed ${type}` }] : [];
}

function wordingVotes(titleSignals) {
  if (titleSignals.conventionalType) {
    return [];
  }
  if (titleSignals.bug) {
    return [{ source: "wording", kind: "fix", detail: "the title describes a fix" }];
  }
  if (titleSignals.enhancement) {
    return [{ source: "wording", kind: "feature", detail: "the title describes a feature" }];
  }
  return [];
}

function collectReleaseImpactVotes({ title, body, headRef, commits, files, titleSignals, changesTests, uiCapabilityFiles }) {
  const commitVotes = (commits || []).flatMap((commit, index) => {
    const message = String(commit?.message || "");
    const detail = `the commit "${message.split("\n")[0].trim()}"`;
    return [...conventionalVotes("commit", message, detail), ...footerVotes("commit", message, detail)];
  });
  return [
    ...conventionalVotes("title", title, "the title"),
    ...footerVotes("body", body, "the description"),
    ...commitVotes,
    ...changelogVotes(files || []),
    ...branchVotes(headRef),
    ...wordingVotes(titleSignals || {}),
    ...(changesTests ? [{ source: "tests", kind: "change", detail: "tests change together with the source" }] : []),
    ...((uiCapabilityFiles || []).length > 0
      ? [{ source: "structure", kind: "feature", detail: `${uiCapabilityFiles.slice(0, 3).join(", ")} add${uiCapabilityFiles.length === 1 ? "s" : ""} new interface functions or classes` }]
      : [])
  ];
}

function strongestKind(votes) {
  return votes.reduce((best, vote) => (best === null || KIND_RANK[vote.kind] > KIND_RANK[best] ? vote.kind : best), null);
}

function maxDecision(left, right) {
  return DECISION_RANK[right] > DECISION_RANK[left] ? right : left;
}

function resolveReleaseImpact(input) {
  const votes = collectReleaseImpactVotes(input);
  const releaseVote = (vote) => vote.kind !== "internal";
  const authoritative = votes.filter((vote) => DECLARED_SOURCES.has(vote.source) || RECORDED_SOURCES.has(vote.source));
  const authoritativeRelease = authoritative.filter(releaseVote);
  const declaresInternal = authoritative.some((vote) => vote.kind === "internal");
  const inferredRelease = votes.filter((vote) => !authoritative.includes(vote) && releaseVote(vote));

  let deciding = authoritativeRelease;
  if (deciding.length === 0 && !declaresInternal) {
    deciding = inferredRelease;
  }
  const kind = deciding.length > 0 ? strongestKind(deciding) : declaresInternal ? "internal" : "unclassified";
  const decidingVotes = deciding.filter((vote) => vote.kind === kind);
  const conflicts = declaresInternal && authoritativeRelease.length > 0
    ? [`the declared internal type is overruled because ${summarizeEvidence(authoritativeRelease).join("; ")}`]
    : [];

  const ships = Boolean(input.shipsBehavior);
  const structural = input.structuralSemver || { decision: "none", hardRule: false, hardSignals: [] };
  const structuralCounts = Boolean(input.structuralReleaseRelevant) || Boolean(structural.hardRule);
  const kindDecision = ships ? DECISION_BY_KIND[kind] || "none" : "none";
  const structuralDecision = structuralCounts ? String(structural.decision || "none") : "none";
  const decision = maxDecision(structuralDecision, kindDecision);

  const kindHardSignals = ships && kindDecision !== "patch" && kindDecision !== "none"
    ? [...new Set(decidingVotes.map((vote) => HARD_SIGNAL_BY_SOURCE_KIND[`${vote.source}:${vote.kind}`]).filter(Boolean))]
    : [];
  const hardSignals = [...new Set([...(structural.hardSignals || []), ...kindHardSignals])];
  const semver = {
    ...structural,
    decision,
    hardRule: Boolean(structural.hardRule) || kindHardSignals.length > 0,
    hardSignals
  };

  return {
    conflicts,
    decidingVotes,
    kind,
    kindDecision,
    label: ships && kindDecision !== "none" ? KIND_LABELS[kind] || "" : "",
    semver,
    ships,
    votes
  };
}

const MAX_LISTED_COMMITS = 2;

function summarizeEvidence(votes) {
  const details = [...new Set(votes.filter((vote) => vote.source !== "commit").map((vote) => vote.detail))];
  const commits = [...new Set(votes.filter((vote) => vote.source === "commit").map((vote) => vote.detail))];
  const listed = commits.slice(0, MAX_LISTED_COMMITS);
  const more = commits.length - listed.length;
  return [
    ...listed,
    ...(more > 0 ? [`${more} more commit${more === 1 ? " says" : "s say"} the same`] : []),
    ...details
  ];
}

function describeReleaseImpact(impact) {
  const evidence = summarizeEvidence(impact.decidingVotes);
  return [
    `Change kind: ${impact.kind}.`,
    ...(evidence.length > 0 ? [`Why: ${evidence.join("; ")}.`] : []),
    ...impact.conflicts.map((conflict) => `Conflict: ${conflict}.`)
  ];
}

module.exports = {
  CHANGELOG_PATH_PATTERN,
  collectReleaseImpactVotes,
  describeReleaseImpact,
  summarizeEvidence,
  resolveReleaseImpact
};
