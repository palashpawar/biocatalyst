# Deploying the board

The site is static: `web/index.html` plus `web/board.json`. Nothing runs at
request time, so there is no bundle limit, no cold start, and no function
timeout to design around.

## Why the refresh does not run on Vercel

A full refresh takes roughly ten minutes — SEC throttles the EDGAR leg and
there are ~150 option chains to price. Vercel functions cap at 60s on Hobby and
300s on Pro, and Cron jobs inherit that. No configuration makes it fit.

Instead GitHub Actions runs the pipeline nightly (6-hour ceiling, free), commits
`web/board.json`, and Vercel redeploys on push. No Vercel CLI or deploy token.

## One-time setup

```bash
git init && git add -A && git commit -m "initial commit"
gh repo create biocatalyst --private --source=. --push
gh secret set BIOCATALYST_SEC_UA --body "biocatalyst research <your-email>"
```

Then on vercel.com: **Add New → Project → import the repo**. Framework preset
*Other*; `vercel.json` already sets the output directory to `web`. Deploy.

Trigger the first data run:

```bash
gh workflow run "Refresh catalyst board"
```

## Local loop

```bash
python3 -m biocatalyst.cli refresh   # pull every source
python3 -m biocatalyst.cli export    # write web/board.json
cd web && python3 -m http.server 5099
```

## Things that will eventually bite

- **Yahoo may throttle cloud IPs.** Prices and option chains come from
  yfinance; datacenter ranges get rate-limited more aggressively than a home
  connection. If the Actions run degrades, the export sanity-check fails the
  job rather than publishing a thin board.
- **SEC wants a real contact string** in `BIOCATALYST_SEC_UA`. Without it they
  throttle harder and may block.
- **GitHub's cron is best-effort** on free runners and can be delayed at peak
  times. It is not a market-open guarantee.
- The DuckDB file stays local and gitignored; only the JSON is published.
