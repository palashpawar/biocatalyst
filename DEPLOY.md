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

## Why Vercel runs the clock

GitHub's own `schedule` trigger never fired for this repo. Two cron
expressions, zero `schedule` events, with Actions enabled, the workflow valid
and on the default branch, and the repo neither forked nor archived. GitHub
documents scheduled workflows as best-effort and they can be dropped entirely.

So Vercel Cron keeps the time and GitHub still does the work. `api/refresh.js`
makes a single `workflow_dispatch` call; the ~8 minute refresh runs on Actions,
where there is no function timeout to fight.

Two environment variables are needed on the Vercel project:

| Variable | What |
|---|---|
| `GH_DISPATCH_TOKEN` | Fine-grained PAT, **Actions: read and write**, scoped to this repo only |
| `CRON_SECRET` | Any random string. Vercel sends it as a bearer token on cron invocations, and the function rejects anything else — without it the endpoint is a public button that starts a job in your repo. |

Verify a dispatch by hand once deployed:

```bash
curl -s -X GET "https://<your-app>.vercel.app/api/refresh" \
  -H "Authorization: Bearer $CRON_SECRET"
# {"ok":true,"dispatched":"palashpawar/biocatalyst@master", ...}
gh run list --limit 1
```

Note that Hobby-plan cron jobs run once per day at an approximate time, which
is fine for a daily board.

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
