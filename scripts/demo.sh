#!/usr/bin/env bash
# Walks through the API of the local stack: a search, a triage of an incident that the corpus
# documents and a triage of one that it does not.
#
# With TRIAGE_LLM=replay the answers are the recorded ones of the evaluation, so this script
# also checks them and fails if either is not what it should be. Against the live API the
# answers are printed and not checked: a model may word them differently on each call.
set -euo pipefail

base=${TRIAGE_URL:-http://127.0.0.1:8088}
data=$(cd "$(dirname "$0")/.." && pwd)/data

incident() {
  jq --compact-output --arg id "$1" 'select(.id == $id) | del(.id)' "$data/incidents.jsonl"
}

post() {
  curl --silent --show-error --fail-with-body --header 'content-type: application/json' \
    --data "$2" "$base$1"
}

health=$(curl --silent --show-error --fail "$base/healthz")
mode=$(jq --raw-output .llm <<<"$health")
echo "== The API is up: model $(jq --raw-output .model <<<"$health"), answers from: $mode"

echo
echo "== Search: the three documents closest to one log line"
post /v1/search '{"query": "INV-5611 unmapped location code count=4312", "limit": 3}' |
  jq --raw-output '.[] | "\(.rank). \(.id)  \(.title)"'

echo
echo "== Triage of T10: thousands of products at zero stock, which the corpus documents"
with_evidence=$(post /v1/triage "$(incident T10)")
jq '{decision: .triage.decision, category: .triage.cause_category, cause: .triage.likely_cause,
     citations: .triage.citations, next_step: .triage.next_step, cost_usd, seconds}' <<<"$with_evidence"

echo
echo "== Triage of T06: timeouts in a service that the corpus does not document"
without_evidence=$(post /v1/triage "$(incident T06)")
jq '{decision: .triage.decision, category: .triage.cause_category,
     next_step: .triage.next_step, retrieved: [.evidence[].id]}' <<<"$without_evidence"

if [ "$mode" = replay ]; then
  jq --exit-status '.triage.decision == "triage" and .triage.cause_category == "config_change"
    and (.triage.citations | length) > 0' <<<"$with_evidence" >/dev/null ||
    { echo "T10 should be triaged as a configuration change, with a citation" >&2; exit 1; }
  jq --exit-status '.triage.decision == "insufficient_evidence" and (.triage.citations | length) == 0' \
    <<<"$without_evidence" >/dev/null ||
    { echo "T06 should be answered with insufficient evidence" >&2; exit 1; }
  echo
  echo "== Both recorded answers are what they should be."
fi
