const { scoreDeterministicEvidence } = require("./measurement/scorer.cjs");
const { applyLabelWordBudget } = require("./measurement/label_selection.cjs");
const { AutobotLabelRegistry } = require("./labels/registry.cjs");
const { VULNERABILITY_CLASS_TEXT_PATTERN } = require("./constants.cjs");

const MAX_LABEL_WORDS = 2;
const BUG_TITLE_PATTERN = /\b(bugs?|crash(?:es|ed|ing)?|errors?|fail(?:s|ed|ing|ures?)|broken|regressions?|freez(?:e|es|ing)|froze|hang(?:s|ing)?|stuck|unresponsive|(?:does ?n[o']?t|doesn't|not) work(?:ing)?|incorrect(?:ly)?|wrong)\b/;
const NEGATED_BUG_PATTERN = /\b(?:not|isn'?t|is not|no|never)\s+(?:an?\s+)?(?:bugs?|regressions?|defects?)\b/;
const TRACEBACK_PATTERN = /\btraceback \(most recent call last\)/;
const CRASH_PATTERN = /\bcrash(?:es|ed|ing)?\b/;
const VULNERABILITY_KEYWORD_PATTERN = /\b(vulnerab\w*|advisory|cve-\d{4}-\d+|ghsa-[a-z0-9-]+|exploit(?:able|s|ed)?|security advisory)\b/;
const FEATURE_REQUEST_PATTERN = /\b(feature request|enhancement|please (?:add|support|allow|make|let|include|provide|implement|consider)|it would be (?:great|nice|useful|helpful|good|awesome)|(?:would|i'?d) (?:love|like) (?:to (?:have|see|be able)|a|an|the|support)|i wish|could you (?:add|support|make|allow)|can you (?:add|support|make|allow)|add support for|should be able to|is there a way to|it would help)\b/;
const BEHAVIOR_REQUEST_FALLBACK_LABEL = "runtime";
const SECURITY_REPORT_FALLBACK_LABEL = "vulnerability";

class AutobotIssueClassifier {
  static ISSUE_RELEASE_RELEVANT_LABELS = new Set([...AutobotLabelRegistry.RELEASE_CRITICAL_LABELS]);

  static TEMPLATE_MARKERS = Object.freeze({
    documentation: Object.freeze([
      "documentation issue report",
      "link(s) to the affected documentation",
      "detailed description of the problem",
      "proposed solution (optional)"
    ]),
    bug: Object.freeze([
      "thank you for helping us squash this bug",
      "detailed steps to reproduce",
      "potential causes / workarounds / related issues (optional)",
      "custom modifications / configuration"
    ]),
    feature: Object.freeze([
      "proposing an improvement or enhancement",
      "current situation and problem/opportunity",
      "proposed improvement/enhancement",
      "potential costs, challenges, and considerations",
      "alternatives considered (optional)",
      "proposed steps or implementation plan (optional)"
    ])
  });

  static normalizeIssue(issue) {
    const title = String(issue?.title || "").trim();
    const body = String(issue?.body || "").trim();
    const text = `${title}\n${body}`.toLowerCase();
    return {
      title,
      body,
      text,
      normalizedTitle: title.toLowerCase(),
      normalizedBody: body.toLowerCase()
    };
  }

  static hasAnyMarker(text, markers) {
    return markers.some((marker) => text.includes(marker));
  }

  static pushUnique(items, value) {
    if (value && !items.includes(value)) {
      items.push(value);
    }
  }

  static pushEvidenceItem(items, ruleId) {
    if (ruleId && !items.some((item) => item.ruleId === ruleId)) {
      items.push({ ruleId });
    }
  }

  static scoredLabelNames(entries) {
    return (entries || []).map((entry) => String(entry?.label || "").trim()).filter(Boolean);
  }

  static hasScoredLabel(scoring, label, flagName = "emitted") {
    return Boolean(scoring?.labelScores?.[label]?.[flagName]);
  }

  static hasLabelOrAncestor(labels, expectedLabel) {
    return (labels || []).some((label) => {
      const normalizedLabel = AutobotLabelRegistry.normalizeLabelName(label);
      return normalizedLabel === expectedLabel || AutobotLabelRegistry.isDescendantLabel(normalizedLabel, expectedLabel);
    });
  }

  static deriveTechnicalLabels(normalized) {
    const labels = [];
    const text = normalized.text;

    if (/readme/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "readme");
    }
    if (/docs?\/reference|openapi|swagger|api docs|reference docs/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "api doc");
    }
    if (/architecture|module map|data flow/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "architecture doc");
    }
    if (/install|getting started|quickstart/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "install guide");
      if (/quickstart|getting started/.test(text)) {
        AutobotIssueClassifier.pushUnique(labels, "quickstart");
      }
    }
    if (/troubleshoot|troubleshooting|faq/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "troubleshoot guide");
      if (/faq/.test(text)) {
        AutobotIssueClassifier.pushUnique(labels, "faq");
      }
    }
    if (/pyproject/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "pyproject");
    }
    if (/package-lock|poetry\.lock|pnpm-lock|yarn\.lock|lockfile/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "lockfile");
    }
    if (/codeowners/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "codeowners");
    }
    if (/issue form|issue template/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "issue form");
    }
    if (/pull request template|pr template/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "pull template");
    }
    if (/dependabot/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "dependabot");
    }

    if (/workflow|github actions/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "github actions");
      if (/matrix/.test(text)) {
        AutobotIssueClassifier.pushUnique(labels, "matrix job");
      }
    }
    if (/unit test/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "unit test");
    }
    if (/integration test|scenario|regression/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "integration test");
    }

    if (/\b(apis?|endpoints?|routes?|router|handlers?)\b/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "route");
      if (/path param|route param/.test(text)) {
        AutobotIssueClassifier.pushUnique(labels, "route param");
      }
      if (/query param|querystring|search param/.test(text)) {
        AutobotIssueClassifier.pushUnique(labels, "query param");
      }
    }
    if (/graphql/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "graphql");
      if (/resolver/.test(text)) {
        AutobotIssueClassifier.pushUnique(labels, "resolver");
      }
    }
    if (/webhook/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "webhook");
    }
    if (/request body|request payload|input schema/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "request body");
    }
    if (/response body|response payload|output schema/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "response body");
    }
    if (/validation|validator|pydantic/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "body validation");
    }
    if (/openapi|swagger/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "openapi spec");
    }

    if (/jwt/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "jwt");
    }
    if (/refresh token/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "refresh token");
    }
    if (/rbac|role-based|role mapping/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "rbac");
    }
    if (/oauth|scope/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "oauth scope");
    }
    if (/csrf/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "csrf");
    }
    if (/authz|authorization/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "authz");
    }
    if (VULNERABILITY_KEYWORD_PATTERN.test(text) || VULNERABILITY_CLASS_TEXT_PATTERN.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "vulnerability");
    }
    if (/\b(compliance|soc ?2|pci(?:-dss)?|gdpr|hipaa|iso ?27001|nist|fedramp)\b/.test(text)
      || /\bsarbanes[-\s]+oxley\b/.test(text)
      || /\bsox\s+(?:compliance|controls?|control matrix|attestation|audit)\b/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "compliance");
    }
    if (/\b(hardening|harden|least privilege|defense in depth|deny by default|sandbox|mitigation|sanitize|sanitiz)\b/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "hardening");
    }
    if (/\b(pen[\s-]?test|penetration test|red team)\b/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "pen-test");
    }

    if (/migration|alembic/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "migration file");
      if (/destructive|drop column|drop table/.test(text)) {
        AutobotIssueClassifier.pushUnique(labels, "destructive migration");
      }
      if (/add column|add table|create table/.test(text)) {
        AutobotIssueClassifier.pushUnique(labels, "additive migration");
      }
    }
    if (/json schema/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "json schema");
    }
    if (/proto|protobuf/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "proto schema");
    }
    if (/query|sql|database|postgres|mysql|sqlite/.test(text)) {
      if (/select|read/.test(text)) {
        AutobotIssueClassifier.pushUnique(labels, "read query");
      }
      if (/insert|update|delete|write|upsert/.test(text)) {
        AutobotIssueClassifier.pushUnique(labels, "write query");
      }
    }

    if (/python/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "python version");
    }
    if (/windows/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "windows");
      if (/path separator|\\\\|ntpath|posixpath|pathlib|normpath|realpath/.test(text)) {
        AutobotIssueClassifier.pushUnique(labels, "path separator");
        AutobotIssueClassifier.pushUnique(labels, "path normalization");
      }
      if (/shell command|cmd(?:\.exe)?\s*\/[ck]|powershell(?:\.exe)?|bash\s+-[cl]|\/bin\/(?:sh|bash)/.test(text)) {
        AutobotIssueClassifier.pushUnique(labels, "shell command");
      }
    }
    if (/linux|ubuntu/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "linux");
    }
    if (/macos/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "macos");
    }
    if (/cuda|nvidia/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "cuda");
    }
    if (/env var|environment variable|getenv|dotenv/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "env var");
    }
    if (/filesystem|file system|path separator|filepath|pathlib|normpath|realpath/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "filesystem");
      AutobotIssueClassifier.pushUnique(labels, "path normalization");
    }
    if (/subprocess|spawn|fork|exec\(|child process|child_process|popen|stdout|stderr|stdin|pipe/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "process");
      AutobotIssueClassifier.pushUnique(labels, "subprocess io");
    }
    if (/memory|heap|buffer/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "heap usage");
    }
    if (/cache/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "cache hit");
    }
    if (/latency|cold start|import time/.test(text)) {
      AutobotIssueClassifier.pushUnique(labels, "latency");
    }

    return AutobotIssueClassifier.normalizeOutputLabels(labels);
  }

  static normalizeOutputLabels(labels, options = {}) {
    const maxWords = Math.max(Number(options.maxWords) || MAX_LABEL_WORDS, 1);
    const limit = options.limit;
    return AutobotLabelRegistry.technicalLabelsOnly(applyLabelWordBudget(labels, { maxWords }), limit === undefined ? {} : { limit });
  }

  static analyzeIssueIntake(issue) {
    const normalized = AutobotIssueClassifier.normalizeIssue(issue);
    const evidenceSignals = [];
    const likelyClassification = [];
    const technicalLabels = AutobotIssueClassifier.deriveTechnicalLabels(normalized);

    const documentationTemplate = AutobotIssueClassifier.hasAnyMarker(normalized.text, AutobotIssueClassifier.TEMPLATE_MARKERS.documentation);
    const bugTemplate = AutobotIssueClassifier.hasAnyMarker(normalized.text, AutobotIssueClassifier.TEMPLATE_MARKERS.bug);
    const featureTemplate = AutobotIssueClassifier.hasAnyMarker(normalized.text, AutobotIssueClassifier.TEMPLATE_MARKERS.feature);

    const documentationSignal = documentationTemplate || /\b(docs?|documentation|readme)\b/.test(normalized.normalizedTitle);
    const bugMention = BUG_TITLE_PATTERN.test(normalized.normalizedTitle) || TRACEBACK_PATTERN.test(normalized.normalizedBody);
    const bugSignal = bugTemplate || (bugMention && !NEGATED_BUG_PATTERN.test(normalized.normalizedTitle));
    const enhancementSignal = featureTemplate || FEATURE_REQUEST_PATTERN.test(normalized.text);
    const securityReportSignal = VULNERABILITY_KEYWORD_PATTERN.test(normalized.text) || VULNERABILITY_CLASS_TEXT_PATTERN.test(normalized.text);
    const proposalSignal = /\b(proposal|rfc|request for comments|design proposal|roadmap|future work)\b/.test(normalized.normalizedTitle)
      || /\brequest for comments\b/.test(normalized.text);
    const improvementSignal = /\b(improvement|improve|streamline|simplify|quality of life|qol)\b/.test(normalized.normalizedTitle)
      || /\b(pain point|gap closure|quality of life)\b/.test(normalized.text);
    const runtimeSignal = /\b(runtime|windows|linux|macos|cuda|python)\b/.test(normalized.text);
    const apiSignal = /\b(apis?|endpoints?|webhooks?|http (?:request|response)s?|rest api)\b/.test(normalized.text);

    if (documentationTemplate) {
      AutobotIssueClassifier.pushUnique(evidenceSignals, "Structured documentation issue template fields are present.");
    }
    if (bugTemplate) {
      AutobotIssueClassifier.pushUnique(evidenceSignals, "Structured bug-report fields are present.");
    }
    if (featureTemplate) {
      AutobotIssueClassifier.pushUnique(evidenceSignals, "Structured feature-request fields are present.");
    }
    if (runtimeSignal) {
      AutobotIssueClassifier.pushUnique(evidenceSignals, "The intake mentions runtime or platform context.");
    }
    if (apiSignal) {
      AutobotIssueClassifier.pushUnique(evidenceSignals, "The intake mentions API context.");
    }

    const fallbackTechnicalLabels = [];
    if (documentationSignal) {
      AutobotIssueClassifier.pushUnique(likelyClassification, "The intake describes a documentation surface.");
    } else if (bugSignal) {
      AutobotIssueClassifier.pushUnique(likelyClassification, "The intake describes a concrete defect or regression.");
    } else if (enhancementSignal || proposalSignal || improvementSignal) {
      AutobotIssueClassifier.pushUnique(likelyClassification, "The intake asks for expanded or improved behavior.");
      if (proposalSignal) {
        AutobotIssueClassifier.pushUnique(likelyClassification, "The request is framed as a proposal or future design direction.");
      } else if (improvementSignal) {
        AutobotIssueClassifier.pushUnique(likelyClassification, "The request is framed as a targeted improvement or gap closure.");
      }
    }

    if (documentationSignal && !AutobotIssueClassifier.hasLabelOrAncestor(technicalLabels, "docs")) {
      AutobotIssueClassifier.pushUnique(fallbackTechnicalLabels, "guide");
    }
    if (runtimeSignal && !AutobotIssueClassifier.hasLabelOrAncestor(technicalLabels, "runtime")) {
      AutobotIssueClassifier.pushUnique(fallbackTechnicalLabels, "platform");
    }
    if (apiSignal && !AutobotIssueClassifier.hasLabelOrAncestor(technicalLabels, "api")) {
      AutobotIssueClassifier.pushUnique(fallbackTechnicalLabels, "route");
    }
    if (securityReportSignal && !AutobotIssueClassifier.hasLabelOrAncestor(technicalLabels, "security posture")) {
      AutobotIssueClassifier.pushUnique(fallbackTechnicalLabels, SECURITY_REPORT_FALLBACK_LABEL);
    }
    if (
      (enhancementSignal || proposalSignal || improvementSignal || bugSignal)
      && !documentationSignal
      && technicalLabels.length === 0
      && fallbackTechnicalLabels.length === 0
    ) {
      AutobotIssueClassifier.pushUnique(fallbackTechnicalLabels, BEHAVIOR_REQUEST_FALLBACK_LABEL);
    }
    const crashSignal = CRASH_PATTERN.test(normalized.normalizedTitle) || TRACEBACK_PATTERN.test(normalized.normalizedBody);
    if (bugSignal && crashSignal && !AutobotIssueClassifier.hasLabelOrAncestor(technicalLabels, "process")) {
      AutobotIssueClassifier.pushUnique(fallbackTechnicalLabels, "process");
    }

    if (documentationSignal && /\bproblem|incorrect|missing|unclear|outdated\b/.test(normalized.text)) {
      AutobotIssueClassifier.pushUnique(evidenceSignals, "The intake describes a documentation defect or gap.");
    }
    if (bugSignal) {
      AutobotIssueClassifier.pushUnique(evidenceSignals, "The intake explicitly describes incorrect behavior or a failure.");
    }
    if (enhancementSignal || proposalSignal || improvementSignal) {
      AutobotIssueClassifier.pushUnique(evidenceSignals, "The intake asks for new or improved behavior rather than reporting a confirmed defect.");
    }
    if (proposalSignal) {
      AutobotIssueClassifier.pushUnique(evidenceSignals, "The intake frames the request as a proposal or future design direction.");
    }
    if (improvementSignal && !proposalSignal) {
      AutobotIssueClassifier.pushUnique(evidenceSignals, "The intake frames the request as a targeted improvement or gap closure.");
    }

    const evidenceItems = [];
    if (documentationSignal) {
      AutobotIssueClassifier.pushEvidenceItem(evidenceItems, "issue-documentation-report");
    } else if (bugSignal) {
      AutobotIssueClassifier.pushEvidenceItem(evidenceItems, "issue-bug-report");
    } else if (enhancementSignal || proposalSignal || improvementSignal) {
      AutobotIssueClassifier.pushEvidenceItem(evidenceItems, "issue-enhancement-request");
      if (proposalSignal) {
        AutobotIssueClassifier.pushEvidenceItem(evidenceItems, "issue-proposal-request");
      } else if (improvementSignal) {
        AutobotIssueClassifier.pushEvidenceItem(evidenceItems, "issue-improvement-request");
      }
    }
    if (runtimeSignal) {
      AutobotIssueClassifier.pushEvidenceItem(evidenceItems, "issue-runtime-context");
    }
    if (apiSignal) {
      AutobotIssueClassifier.pushEvidenceItem(evidenceItems, "issue-api-context");
    }

    const deterministicScoring = scoreDeterministicEvidence({
      evidenceItems,
      options: {
        maxWords: MAX_LABEL_WORDS
      }
    });
    const deterministicLabels = AutobotIssueClassifier.normalizeOutputLabels([
      ...AutobotIssueClassifier.scoredLabelNames(deterministicScoring.emittedLabels),
      ...technicalLabels,
      ...fallbackTechnicalLabels
    ]);
    const deterministicPrimaryLabels = AutobotIssueClassifier.normalizeOutputLabels([
      ...AutobotIssueClassifier.scoredLabelNames(deterministicScoring.primaryLabels),
      ...technicalLabels,
      ...fallbackTechnicalLabels
    ]);
    const outputLabels = AutobotIssueClassifier.normalizeOutputLabels([
      ...technicalLabels,
      ...fallbackTechnicalLabels,
      ...deterministicPrimaryLabels,
      ...deterministicLabels
    ]);

    return {
      title: normalized.title,
      body: normalized.body,
      text: normalized.text,
      labels: outputLabels,
      evidenceSignals,
      evidenceItems,
      likelyClassification,
      deterministicLabels,
      deterministicPrimaryLabels,
      deterministicSemver: deterministicScoring.semver,
      releaseRelevant: outputLabels.some((label) => AutobotLabelRegistry.isReleaseRelevantLabel(label))
        || deterministicLabels.some((label) => AutobotIssueClassifier.ISSUE_RELEASE_RELEVANT_LABELS.has(label))
    };
  }
}

function analyzeIssueIntake(issue) {
  return AutobotIssueClassifier.analyzeIssueIntake(issue);
}

module.exports = {
  AutobotIssueClassifier,
  analyzeIssueIntake
};
