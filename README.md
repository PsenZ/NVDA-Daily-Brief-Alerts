# NVDA Daily Brief

Automated daily NVDA brief (technical, fundamentals, news catalysts, options, risk) sent by email at 07:30 Sydney time.

## Setup

1. Create a Yahoo App Password for your account.
2. Add GitHub Actions Secrets in this repo:
   - `SMTP_USER` = your Yahoo email (e.g. `zhengps000916@yahoo.com`)
   - `SMTP_APP_PASSWORD` = Yahoo App Password
   - `FROM_EMAIL` = sender email (same as Yahoo account)
   - `TO_EMAIL` = recipient email

The workflow runs every 20 minutes and only sends within the 07:30 ±10 minutes Sydney time window.

## Manual run

Use GitHub Actions `workflow_dispatch` to run on demand.
