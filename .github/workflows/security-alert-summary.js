module.exports = async ({ github, context, core }) => {
const COMMENT_MARKER = process.env.COMMENT_MARKER;
const SEVERITY_COLUMNS = ['critical', 'high'];
const HIGH_CRITICAL_SEVERITIES = new Set(SEVERITY_COLUMNS);
const { owner, repo } = context.repo;
const repositoryName = `${owner}/${repo}`;
const encodeFilterQuery = (filter) =>
  encodeURIComponent(filter).replace(/%20/g, '+');
const dependabotHighCriticalFilter =
  'is:open severity:critical severity:high';
const codeqlHighCriticalFilter =
  'is:open tool:CodeQL severity:critical severity:high';
const dependabotAlertsUrl =
  `${context.serverUrl}/${owner}/${repo}/security/dependabot` +
  `?q=${encodeFilterQuery(dependabotHighCriticalFilter)}`;
const codeqlAlertsUrl =
  `${context.serverUrl}/${owner}/${repo}/security/code-scanning` +
  `?q=${encodeFilterQuery(codeqlHighCriticalFilter)}`;

const normalizeSeverity = (value) =>
  (value || '').toString().trim().toLowerCase();

const emptySeverityCounts = () =>
  Object.fromEntries(SEVERITY_COLUMNS.map((s) => [s, 0]));

const countBySeverity = (alerts, getSeverity) => {
  const counts = emptySeverityCounts();
  for (const alert of alerts) {
    const severity = normalizeSeverity(getSeverity(alert));
    if (HIGH_CRITICAL_SEVERITIES.has(severity)) {
      counts[severity] += 1;
    }
  }
  return counts;
};

const sumCounts = (...countObjects) => {
  const total = emptySeverityCounts();
  for (const counts of countObjects) {
    for (const severity of SEVERITY_COLUMNS) {
      total[severity] += counts[severity] || 0;
    }
  }
  return total;
};

const rowTotal = (counts) =>
  SEVERITY_COLUMNS.reduce((sum, severity) => sum + counts[severity], 0);

const formatCountsRow = (label, counts) => {
  const cells = SEVERITY_COLUMNS.map((s) => counts[s] || 0);
  const total = rowTotal(counts);
  return `| ${label} | ${cells.join(' | ')} | ${total} |`;
};

const isCodeqlAlert = (alert) => {
  const toolName = (alert.tool?.name || alert.tool_name || '')
    .toString()
    .toLowerCase();
  return toolName === 'codeql' || toolName.includes('codeql');
};

let dependabotAlerts = [];
let codeqlAlerts = [];
const errors = [];

core.info(`Checking security alerts for ${repositoryName}...`);

try {
  dependabotAlerts = await github.paginate(
    github.rest.dependabot.listAlertsForRepo,
    { owner, repo, state: 'open', severity: 'critical,high', per_page: 100 }
  );
} catch (error) {
  const message = `Dependabot alerts: ${error.message}`;
  errors.push(message);
  core.warning(message);
  core.warning(
    'Ensure the SAST GitHub App has Dependabot alerts (Read) on this repository.'
  );
}

try {
  const scanningAlerts = await github.paginate(
    github.rest.codeScanning.listAlertsForRepo,
    { owner, repo, state: 'open', per_page: 100 }
  );
  codeqlAlerts = scanningAlerts.filter(isCodeqlAlert);
} catch (error) {
  const message = `CodeQL / code scanning alerts: ${error.message}`;
  errors.push(message);
  core.warning(message);
  core.warning(
    'Ensure the SAST GitHub App has Code scanning alerts (Read) or security-events access.'
  );
}

const dependabotCounts = countBySeverity(
  dependabotAlerts,
  (alert) => alert.security_advisory?.severity
);
const codeqlCounts = countBySeverity(
  codeqlAlerts,
  (alert) =>
    alert.rule?.security_severity_level || alert.rule?.severity
);
const totalCounts = sumCounts(dependabotCounts, codeqlCounts);
const totalOpenAlerts = rowTotal(totalCounts);

const buildSummaryTable = () => {
  const header = '| Type | Critical | High | Total |';
  const separator = '| --- | ---: | ---: | ---: |';
  return [
    header,
    separator,
    formatCountsRow('Dependabot', dependabotCounts),
    formatCountsRow('CodeQL', codeqlCounts),
    formatCountsRow('**Total**', totalCounts),
  ].join('\n');
};

const buildCommentBody = () => {
  const lines = [
    COMMENT_MARKER,
    '## Security Alert Summary',
    '',
    `Below is a summary of all open critical and high alerts for **${repositoryName}**. To help maintain a secure environment, please prioritize reviewing and resolving these alerts at your earliest convenience.`,
    '',
    'You can find all open alerts at the following links:',
    '',
    `- [Dependabot alerts (open, critical & high)](${dependabotAlertsUrl})`,
    `- [CodeQL alerts (open, critical & high)](${codeqlAlertsUrl})`,
    '',
    'Your help is greatly appreciated in keeping our code secure.',
    '',
    buildSummaryTable(),
  ];

  if (errors.length > 0) {
    lines.push('', '### API warnings', '', ...errors.map((e) => `- ${e}`));
  }

  return lines.join('\n');
};

const commentBody = buildCommentBody();
const summaryTable = buildSummaryTable();

core.summary.addHeading('Security Alert Summary');
core.summary.addRaw(
  `Open critical/high alerts for **${repositoryName}**: **${totalOpenAlerts}**`
);
core.summary.addRaw(summaryTable);
if (errors.length > 0) {
  core.summary.addHeading('API warnings', 3);
  core.summary.addRaw(errors.map((e) => `- ${e}`).join('\n'));
}

if (context.eventName === 'pull_request') {
  const issueNumber = context.payload.pull_request.number;

  const { data: comments } = await github.rest.issues.listComments({
    owner,
    repo,
    issue_number: issueNumber,
    per_page: 100,
  });

  const existing = comments.find((c) =>
    (c.body || '').includes(COMMENT_MARKER)
  );

  if (totalOpenAlerts > 0) {
    if (existing) {
      await github.rest.issues.updateComment({
        owner,
        repo,
        comment_id: existing.id,
        body: commentBody,
      });
      core.info(`Updated existing PR comment ${existing.id}`);
    } else {
      await github.rest.issues.createComment({
        owner,
        repo,
        issue_number: issueNumber,
        body: commentBody,
      });
      core.info('Created new PR comment');
    }
  } else if (existing) {
    await github.rest.issues.deleteComment({
      owner,
      repo,
      comment_id: existing.id,
    });
    core.info(`Deleted stale PR comment ${existing.id} (no open alerts)`);
  } else {
    core.info('No open critical/high alerts — skipping PR comment.');
  }
} else {
  core.info(
    'Non-PR event: summary written to job output only (no PR comment).'
  );
}

const dependabotHighCriticalTotal = rowTotal(dependabotCounts);
const codeqlHighCriticalTotal = rowTotal(codeqlCounts);
core.info(
  `Open critical/high alerts: ${totalOpenAlerts} ` +
    `(Dependabot: ${dependabotHighCriticalTotal}, CodeQL: ${codeqlHighCriticalTotal}).`
);
};
