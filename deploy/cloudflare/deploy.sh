#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
test -s ../aws/.api-key
export CLOUDFLARE_EMAIL="${CLOUDFLARE_EMAIL:-jamie@wavey.ai}"
export CLOUDFLARE_API_KEY="${CLOUDFLARE_API_KEY:-$(tr -d '\r\n' < ../../../.cloudflare-token)}"
npx wrangler deploy
npx wrangler secret put STEMS_API_KEY < ../aws/.api-key
