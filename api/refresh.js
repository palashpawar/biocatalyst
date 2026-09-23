// Triggers the GitHub Actions refresh workflow.
//
// GitHub's own `schedule` trigger never fired for this repo -- two different
// cron expressions, zero schedule events, with Actions enabled and the
// workflow valid on the default branch. Vercel's scheduler is reliable, so it
// runs the clock and GitHub still does the work: this function only makes one
// API call, while the ~8 minute refresh stays on Actions where there is no
// timeout to fight.

const OWNER = "palashpawar";
const REPO = "biocatalyst";
const WORKFLOW = "refresh.yml";
const REF = "master";

export default async function handler(req, res) {
  // Vercel signs cron invocations with CRON_SECRET. Without this check the
  // endpoint is a public button that starts a job in someone else's repo.
  const secret = process.env.CRON_SECRET;
  if (secret && req.headers.authorization !== `Bearer ${secret}`) {
    return res.status(401).json({ ok: false, error: "unauthorized" });
  }

  const token = process.env.GH_DISPATCH_TOKEN;
  if (!token) {
    return res.status(500).json({
      ok: false,
      error: "GH_DISPATCH_TOKEN is not set in the Vercel environment",
    });
  }

  const url = `https://api.github.com/repos/${OWNER}/${REPO}` +
              `/actions/workflows/${WORKFLOW}/dispatches`;

  let r;
  try {
    r = await fetch(url, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
        Accept: "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type": "application/json",
        "User-Agent": "biocatalyst-cron",
      },
      body: JSON.stringify({ ref: REF }),
    });
  } catch (err) {
    return res.status(502).json({ ok: false, error: String(err) });
  }

  // A successful dispatch returns 204 with no body.
  if (r.status === 204) {
    return res.status(200).json({
      ok: true,
      dispatched: `${OWNER}/${REPO}@${REF}`,
      at: new Date().toISOString(),
    });
  }

  const body = await r.text().catch(() => "");
  return res.status(502).json({
    ok: false,
    status: r.status,
    // 401/403 means the token is missing the Actions write scope or has
    // expired; 404 usually means it cannot see the repo at all.
    hint: r.status === 404
      ? "token cannot see the repo, or the workflow filename is wrong"
      : r.status === 403 || r.status === 401
        ? "token lacks Actions: read-and-write on this repo, or has expired"
        : undefined,
    body: body.slice(0, 300),
  });
}
