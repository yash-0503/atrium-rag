# The launch checklist — what is left

Written 2026-11-11, after the site went live on Vercel and the
package went to PyPI. Everything below is either a thing you do or a
thing I do. Nothing here is a research question — the research is
done; what is left is execution.

---

## 0. Where things actually stand

| | done | left |
|---|---|---|
| **The library** | ✅ 299 tests pass, ruff clean, twine passes, on PyPI, repo public | the hybrid retrieval work |
| **The site** | ✅ live on Vercel, SEO complete (sitemap, robots, JSON-LD, llms.txt), analytics installed, waitlist modal, security headers | `NEXT_PUBLIC_SITE_URL`, the Resend env vars, the push |
| **The content** | ✅ README rewritten, 12 issues written, GIF recorded, benchmark config printed | the launch posts, the reel, the benchmark page |
| **The marketing** | — | everything |

**The one thing that is broken right now:** the site's canonical URL,
sitemap, robots.txt and `og:url` all say `https://atrium.vercel.app`,
which returns HTTP 400 and serves nothing. Google is being told the
canonical address of your site is a dead host. One environment variable
fixes it. Do it first.

---

## 1. Today — thirty minutes, and one of them is a real bug

### 1.1 The two environment variables

Vercel → your project → **Settings → Environment Variables** → add to
**all three environments** (Production, Preview, Development):

```
NEXT_PUBLIC_SITE_URL=https://atrium-rag.com
BUTTONDOWN_API_KEY=<the key from Buttondown>
```

Then **redeploy** (Deployments → the three dots → Redeploy).

`NEXT_PUBLIC_SITE_URL` is what makes the canonical link, the Open
Graph URL and image, the sitemap, robots.txt, llms.txt and the JSON-LD
all point at the right host. Without it they point at a URL that 400s.

`BUTTONDOWN_API_KEY` is what makes the waitlist collect addresses.
Without it the deployed form answers 503 and the addresses go
nowhere.

### 1.2 Push the site to GitHub

**`atrium-web` still has no git remote.** Vercel is deploying from
your laptop; if you lose it, or a collaborator clones it, there is
nothing. Create the repo on GitHub (public), then:

```bash
cd ~/Desktop/atrium-web
git remote add origin https://github.com/yash-0503/atrium-web.git
git push -u origin main
```

Then in Vercel: **Settings → Git**, connect the repository, and set
the production branch to `main`. After that, every push deploys.

### 1.3 Check the two things that are easy to get wrong

```bash
# The site answers on its own domain
curl -sI https://atrium-rag.com | head -1

# The canonical URL is the domain, not the vercel.app one
curl -s https://atrium-rag.com | grep -o 'rel="canonical" href="[^"]*"'

# The package installs from PyPI, not from your disk
pipx install atrium-rag && atrium version
# or, in a clean venv:
python3 -m venv /tmp/check && /tmp/check/bin/pip install atrium-rag
/tmp/check/bin/atrium version
```

---

## 2. The domain — `atrium-rag.space` vs `atrium-rag.com`

You asked about buying `atrium-rag.space` from Vercel. **Vercel does
not sell domains** — it is a registrar *reseller* for a handful of
TLDs, and `.space` is not one of them. Vercel's Domains page can only
sell what it stocks, and the price is at cost.

**My recommendation: buy `atrium-rag.com` from Cloudflare Registrar
instead.** Reasons:

- `.com` is what people type and what ends up in a tweet. A developer
  who hears "atrium" will type `atrium.com`, `atrium.dev`,
  `atrium-rag.com` — very rarely `atrium-rag.space`.
- Cloudflare Registrar sells at cost with no markup, includes free
  WHOIS privacy and free DNSSEC, and gives you the DNS you need in
  step 3 for free.
- `.space` is a cheap TLD with a spammy reputation in some email
  filters. Not a problem for a docs site, but it is a small tax on a
  domain you will link from everywhere.

If you want `.space` as well, it is cheap — but make it a **301 to the
.com**, never a second site. Two indexable copies of the same content
is the one thing that reliably costs you rankings.

**Where to buy:** Cloudflare Registrar (dashboard → Registrar → Register Domain).
Or Namecheap / Porkbun if you want a simpler UI. Price: ~$10–12/year for
`.com` at Cloudflare.

**Register these two:**
- `atrium-rag.com` — the site
- `atrium-rag.dev` — optional, 301 to the `.com`

**Do not** bother with `atrium-rag.io` or `.ai` — expensive, and the
short `atrium.*` names are all taken (I checked every one with RDAP:
`atrium.io`, `atrium.ai`, `atrium.sh`, `atrium.app`, `atriumcloud.com`,
`getatrium.dev`, `tryatrium.com` are all registered).

---

## 3. DNS and TLS

Cloudflare Registrar includes Cloudflare DNS.

1. **Apex** — Cloudflare's Vercel integration does this for you if you
   add the domain in Vercel first ("Settings → Domains → Add"). Vercel
   then tells you exactly which record to create. Use the **CNAME to
   `cname.vercel-dns.com`** form; it survives an IP change.
2. **`www`** — CNAME to the same, then a **301** to the apex.
3. **`atrium.vercel.app`** — in Vercel, set the production domain to
   `atrium-rag.com` and add a **redirect** from the old one. This is
   the one that stops Google seeing two copies of you.
4. **DNSSEC** — Cloudflare → DNSSEC → Enable. It takes a minute and it
   is a real answer to "is this site trustworthy".
5. **SSL mode** — Cloudflare → SSL/TLS → **Full (strict)**. Not
   "Flexible" — Flexible quietly allows plaintext to the origin.
6. **Security headers** — Vercel → Settings → Security → add:
   - `X-Frame-Options: DENY`
   - `X-Content-Type-Options: nosniff`
   - `Referrer-Policy: strict-origin-when-cross-origin`
   - `Strict-Transport-Security: max-age=63072000; includeSubDomains`
   - A CSP that allows `https://cdn.tailwindcss.com` and your own origin
7. **Search Console** — add the property, verify with DNS, submit
   `https://atrium-rag.com/sitemap.xml`.
8. **Bing Webmaster Tools** — two minutes, and it feeds Copilot and
   Edge answers.

---

## 4. The email setup

**The waitlist now stores addresses in Resend**, not Buttondown, and
the confirmation is sent from your own domain. The route's decision
tree is Resend → Buttondown → Formspree → a local file → a 503 that
says so.

**In Resend:**
1. **Domains → Add Domain** → `atrium-rag.space` → region `us-east-1`.
2. Add the records it gives you at Vercel DNS (four of them — the
   full table is in `DOMAIN_AND_EMAIL.md` §2.3). The **Auto Configure**
   button writes them for you if you would rather. **Strip the domain
   suffix from the Name field** — `send`, not `send.atrium-rag.space`.
3. **Segments → New** → name it `waitlist` → copy the uuid.
4. **API Keys → Create** → copy the key.
5. In Vercel: `RESEND_API_KEY`, `RESEND_SEGMENT_ID`,
   `RESEND_FROM=Atrium <hello@atrium-rag.space>`.

The confirmation email is written in the route — `welcomeEmail` in
`src/app/api/waitlist/route.ts`. Plain HTML, no template, no tracking.
Edit it there.

**Buttondown is now the fallback, not the plan**, and it is worth
keeping for one reason: **Resend's broadcast send is metered
separately, at about $40/month for 5,000 contacts.** Buttondown's free
tier covers a list under a hundred subscribers and its $9/month tier is
cheaper than that. So when there is news to send to the list, use
Buttondown; use Resend for the transactional mail.

If you set up Buttondown as well, its shape is:
`POST https://api.buttondown.email/v1/subscribers` with
`{email_address, type: "regular"}`, and the key comes from **Settings
→ API keys → Generate API key**. In the dashboard: **Emails → New
email → Start from scratch**, subject *"Atrium — you're on the list"*,
type **Regular** (not confirmation — double opt-in means a second
email before the address reaches your list, and for a waitlist you want
it immediately), and the body below.


### The email template

```markdown
Hi,

You're on the Atrium list.

Atrium is a document ingestion library: it parses PDF, Office
documents, HTML, EPUB and scans into structure-aware chunks, embeds
them, and serves retrieval over HTTP. MIT licensed, no GPU, no model
at parse time.

The hosted version — Atrium Cloud — is what this list is for. The
library itself is free and yours today, and it will stay that way.

You'll hear from me when there is something to say. That will mean the
hosted version exists, or that a change landed that you would want to
know about.

If you want to try the library now:

    pip install atrium-rag
    atrium init

— Yash
```

### What you get, and where

- **Buttondown → Subscribers** — the list, the count, and a CSV
  export. This is your "how many people want this" number.
- **Buttondown → Emails → sent** — open rates, click rates.
- The waitlist form on your site posts to the route, which posts to
  Buttondown. The address lands in the Buttondown list, tagged by
  nothing — so add UTM tags to the links you post and use a
  `?utm_source=` on the waitlist links so you can see which platform
  brought people.

### The alternative, if you would rather not use Buttondown

Formspree: create a form, copy the form ID, set
`FORMSPREE_FORM_ID`. Everything else is the same. Buttondown is better
for a waitlist (it is built for one, and it has an export).

---

## 5. What is left to build

### 5.1 The hybrid retrieval work — the one feature that matters

This is issue #4 in `docs/GITHUB_ISSUES.md`, and it is the follow-up
story to the launch. The plan, in the order you approved:

1. **Per-document eligibility flags at ingest.** When a document is
   parsed, record whether it is (a) predominantly Latin script and
   (b) not OCR-derived. Persist them with the document. A scanned PDF
   is Latin characters but wrong ones, so it needs the second flag.
2. **A per-corpus BM25 index.** The lab builds one in RAM at prepare
   time; a server needs one it can consult per query. Start with the
   local store — it is the default and the one the benchmark uses.
   Build at startup, invalidate on write.
3. **Fuse it into `/search`.** The gated implementation already exists
   in `atrium/retrieval.py` and is tested. With ineligible documents
   never entering the index, the confidence gate stops being
   load-bearing — a query on an Arabic document gets dense only
   because the document has no sparse representation, not because a
   heuristic declined.
4. **Re-measure on both suites** — eval_v2 (23 docs, 700 questions) and
   the 19-set (which has the Arabic, scanned and CJK fixtures).
5. **Update the site's number** to whatever that measures.

**The promotion gate:** hybrid ≥ 0.700 on eval_v2 **and** no
regression on the 19-set against dense-only's 0.8158. If the 19-set
regresses, keep hybrid opt-in and publish the dense number as the
headline.

Estimated: 2–4 focused days. This is the thing that lets you publish a
retrieval number the product actually serves.

### 5.2 The pages that earn links

Three pages, in priority order. These are what get you into the
long-tail searches — `docling alternative`, `rag without gpu`,
`chunking strategy comparison` — where the intent is a comparison or a
constraint, and where you can actually rank.

1. **`/docs/benchmark`** — the corpus, the 700 questions, the
   bootstrap, the intervals, the per-document table, and the two
   document types where we lose. This is the page that makes the
   number credible and the page a competitor will link to when they
   disagree with it.
2. **`/docs/what-did-not-work`** — the reranker that regressed on
   forms and OCR'd scans, the hybrid that broke on Arabic, the
   chunking experiment where structure-aware beat everything, the
   confidence gate that turned out to be a wash. **Publishing negative
   results is the single most credible thing you can put next to a
   benchmark table**, and it is the post a senior engineer reads.
3. **`/docs/compare/docling`** (and one per competitor) — where you win
   and where you lose, stated plainly. Two-column papers: we lose.
   Forms: we lose. Parse time and memory: we win by 6× and 3×. A
   comparison page that admits its losses is the one that ranks for
   "alternative"; one that doesn't is the one that gets ignored.

### 5.3 The WebGL deferral — a performance pass

854 KB of HTML and 954 KB of first-load JS, of which ~530 KB is the
WebGL and motion stack. This is your Core Web Vitals, and Core Web
Vitals are a ranking factor.

The page is beautiful and the hero animation is the differentiator, so
**don't remove it — defer it.** `dynamic()`-import the WebGL stage and
mount it after first paint. The text is already prerendered, so your
LCP (the `<h1>`) doesn't wait for the canvas.

Target: under ~200 KB of first-load JS, LCP under 2.0 s on 4G.
Measure before and after. The old Astro site had
`scripts/lighthouse.mjs` — same approach, reusable.

---

## 6. The launch — the marketing

### 6.1 The one claim to lead with

Not "a RAG framework". This:

> **Same retrieval accuracy as Docling. A sixth of the parse time, a
> third of the memory. MIT licensed, no GPU, no model at parse time.**

Every part is measured, and a reader can check it in an afternoon.
Lead on cost, because the accuracy claim is a tie inside a confidence
interval and the cost claim is not close.

### 6.2 The assets you need (one afternoon)

| asset | status |
|---|---|
| The parser GIF | ✅ `docs/images/visualize.gif` — the parser's real output on NIST SP 800-61r2, and `scripts/record_visualizer_gif.py` regenerates it |
| The benchmark chart | ✅ `docs/images/benchmark.png` |
| A **terminal** GIF | ❌ **needed** — `pip install atrium-rag && atrium init && atrium ingest ./docs && atrium search "..."` |
| A **retrieval** GIF | ❌ optional — `/retrieval_lab` showing a query and the passage it found |
| The reel | ❌ **needed** — 30–45 s vertical |

**The terminal GIF is the highest-value asset you do not have.** It
shows the whole product in five seconds, it compresses far better than
screen recordings, and it reads better in a feed. Record it with a
real terminal (`asciinema` or `vhs` — both free) at a readable speed.

**How to record it:**
```bash
# asciinema: cast the shell, then convert to a GIF
asciinema rec demo.cast
asciinema-agg demo.cast demo.gif --fps 20
# or, for a cleaner look, vhs:
vhs demo.cast
chafa -p -f vhs -t 6 demo.cast > demo.gif
```

### 6.3 The reel — the exact structure

30–45 seconds, vertical (1080×1920), **no voice-over, captions
burned in**. Every second accounted for:

| seconds | what | why |
|---|---|---|
| 0–3 | the hook, as text on screen: *"Same accuracy as Docling. 1/6 the parse time. MIT."* | three seconds is all you get before the scroll |
| 3–10 | the benchmark table, filling in | the evidence, immediately |
| 10–22 | the parser GIF (the visualiser clip) | the mechanism — this is the part that makes it feel real |
| 22–30 | the terminal clip: `pip install` → `atrium init` → `atrium search` | the product, in five seconds |
| 30–38 | the benchmark table again, with the cost columns highlighted | the second takeaway |
| 38–45 | `atrium-rag.com` + **"Join the waitlist"** | the conversion |

**The CTA in the last two seconds.** That is the whole conversion
path — a reel with no CTA is a reel that gets views and nothing else.

**Where to post:** LinkedIn (native video gets the most reach with
your audience), X (under 60 s), Instagram Reels, YouTube Shorts. One
cut, four exports, captions burned in.

**One line of text over the whole reel: the number.** The algorithm is
not the story.

### 6.4 The posts — in this order

**Day 1 — Hacker News, 08:00 ET.**

Title: `Show HN: Atrium – MIT-licensed document ingestion, no GPU, same retrieval accuracy as Docling at 1/6 the parse time`

The post body:
1. The one-sentence claim.
2. The benchmark table **with its configuration printed under it**.
3. The reproduction command — `pip install atrium-rag && atrium init`.
4. The cost numbers: 88 s vs 554 s parse time; 1016 MB vs 3482 MB peak.
5. The link to the repo and the benchmark page.
6. **The honest caveats**: two-column papers parse worse; forms lose
   their labels; retrieval is dense-only today, and here is the number
   for that.

**Expect the first comment to be "the README says 0.700, I got 0.6xx".**
Answer it in the thread, in the first hour, with the configuration. The
README and the site both state 0.655 is what `/search` serves — so you
answer with the configuration, not an apology.

**Then:** the site is the link. A reader who lands there sees the
benchmark with its configuration, the limitations, the docs, and the
waitlist.

**Day 1–2 — Reddit.** `r/LocalLLaMA`, `r/MLOps`, `r/datascience`,
`r/opensource`. **Read each subreddit's rules before posting.** The
pattern that works: lead with the measurement, not the product. "We
benchmark at parity with Docling on 29 documents and parse 6× faster
with a third of the memory" gets a technical discussion; "we built a
RAG framework" gets removed.

**Day 2 — Product Hunt.** For launch-day traffic, not for the
technical audience. Cover image: the GIF. Tagline: *"Messy in. Precise
out."* Topics: Open Source, Developer Tools, Artificial Intelligence.

**Day 3 — LinkedIn.** This is where you're reachable. The post is the
*why*:

> *"I benchmarked our document parser against Docling and Unstructured on
> 29 real documents. We were at parity on retrieval accuracy. Then I
> looked at what it cost — 6× the parse time and 3× the memory — and
> rewrote the parser to be pure Python and MIT licensed. No GPU, no model
> at parse time."*

Add: the table, the terminal GIF, the honest caveats, the repo link.

**Day 3–4 — X.** Same claim, threaded, with the GIF. Tag
`@Docling_AI` and `@UnstructuredIO` — they will notice, and it is
better that they hear it from you.

**Day 4 — dev.to / Hashnode.** The technical post: *"How we cut RAG
ingest memory by 3× without giving up accuracy."* The methodology is
the story — RSS, the corpus, the conditions, the CIs. This is the
post that gets saved and shared.

**Day 5 — awesome lists.** `awesome-rag`, `awesome-llmops`,
`awesome-python`. One line each with the benchmark table. Low volume,
high intent.

**Day 7 — a second technical post.** *"We made our sparse ranker
reachable. Here's what it took, and here's what it broke on."* — but
only after §5.1 is done and measured. This is the post the same
audience remembers.

### 6.5 The UTM discipline

Tag every link, from now on:
```
?utm_source=linkedin&utm_medium=post&utm_campaign=launch
?utm_source=x&utm_medium=post&utm_campaign=launch
?utm_source=hn&utm_medium=post&utm_campaign=launch
?utm_source=reddit&utm_medium=post&utm_campaign=launch
?utm_source=producthunt&utm_medium=post&utm_campaign=launch
```

Vercel Analytics → Traffic → Referrers then tells you which platform
brought people. That is the only question that matters, and you
cannot answer it retroactively.

---

## 7. Maintenance, after the launch

- **Re-run the benchmark on every parser or chunker change** and say so
  in the pull request. The published numbers are invalidated by a
  change to either. Issue 9 automates this.
- **Never let a claim and the code drift.** This session found three:
  the README's `auto` claim, the retrieval configuration mismatch, and
  `api.mdx` documenting a route that was deleted. Add a CI check that
  greps the docs for claims the code can verify.
- **Semantic versioning, ruthlessly.** A bug fix is a patch. Removing a
  route is a major. If the 0.1.0 API will change, say so in the
  README before someone builds on it.
- **Reply to issues within a day.** The first five responses set the
  tone more than the first five commits.
- **One thing per pull request**, as `CONTRIBUTING.md` already says.
- **Watch the numbers weekly**: PyPI downloads (pypistats.org),
  Vercel Analytics, Buttondown subscribers, GitHub stars. Four
  numbers. If downloads rise and stars don't, the README is wrong.
  If stars rise and downloads don't, the install path is wrong.

---

## 8. The order, if you want the shortest path to a launch

1. **1.1** — the two environment variables, redeploy. (30 min, and one
   is a real bug.)
2. **1.2** — push `atrium-web` to GitHub and connect it to Vercel.
   (20 min.)
3. **2–3** — buy `atrium-rag.com`, point the apex, 301 `www` and the
   old vercel.app URL, DNSSEC, security headers, Search Console.
   (1 hour.)
4. **4** — the Resend domain, the segment, the three env vars. (30 min.)
5. **5.3** — record the terminal GIF. (30 min.)
6. **6.2–6.3** — cut the reel from the GIFs. (2 hours.)
7. **6.4** — the posts, starting with HN on day 1.
8. **5.1** — the hybrid retrieval work, as the follow-up story.
9. **5.2** — the three pages that earn links, over the following week.

Everything in steps 1–4 is plumbing and takes an afternoon. Step 5–6
is the creative work. Step 7 is the moment. Step 8 is what you do
next.
