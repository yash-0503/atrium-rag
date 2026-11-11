# The domain, the email and the addresses

Written 2026-11-11, after `atrium-rag.space` was registered at Name.com
and pointed at Vercel.

---

## 1. Where the domain stands

| | |
|---|---|
| Domain | `atrium-rag.space`, registered at Name.com, created 2026-11-11, expires 2027-11-11 |
| Nameservers | `ns1.vercel-dns.com` / `ns2.vercel-dns.com` — Vercel's DNS |
| Apex | `atrium-rag.space` → 308 → `https://www.atrium-rag.space/` |
| www | serves the site, HTTP 200 |
| TLS | working, HSTS `max-age=63072000` |
| Registrant data | **redacted** — no name, address or email in public WHOIS or RDAP |

### 1.1 Three things that are still wrong, and all three are one variable

The site's canonical link, `og:url`, sitemap, robots.txt and JSON-LD all
say **`https://atrium.vercel.app`** — a host that returns HTTP 400 and
serves nothing. Google is being told the canonical address of the site
is a dead host.

One environment variable fixes every one of them:

```
Vercel → Project → Settings → Environment Variables
NEXT_PUBLIC_SITE_URL=https://www.atrium-rag.space
```

Add it to **all three environments** (Production, Preview, Development),
then redeploy. Verify:

```bash
curl -s https://www.atrium-rag.space/robots.txt
# should say Sitemap: https://www.atrium-rag.space/sitemap.xml
```

### 1.2 The apex redirect

The apex 308s to `www`. That works, but it is one extra hop and it puts
`www` in every canonical URL. The cleaner arrangement is the other way
round: **the apex serves, and `www` redirects to it**.

In Vercel: **Settings → Domains** → remove `atrium-rag.space` as a
redirect-only domain and add it as a primary domain; then add
`www.atrium-rag.space` as a redirect to the apex.

If you would rather not touch it, the current arrangement is acceptable
— set `NEXT_PUBLIC_SITE_URL=https://www.atrium-rag.space` and be
consistent. Either way the canonical and the domain must agree.

### 1.3 The old Vercel URL

`atrium-web-ashen.vercel.app` currently serves the site too, which means
Google can see two copies. Add a redirect in Vercel from the old
deployment URL to `https://www.atrium-rag.space`. Every page you post
should use the real domain, never the `vercel.app` one.

---

## 2. Email at the domain

You want two different things, and they have different answers.

### 2.1 Sending from the domain — Resend (this is what the waitlist uses)

Resend sends the welcome email and, later, the broadcast to the list.
It puts **no branding in the email body on any plan** — that is why it
replaced Buttondown.

**Setup, once:**

1. Resend → **Domains → Add Domain** → `atrium-rag.space` → region
   `us-east-1`.
2. Resend shows you four records. Create exactly those in **Vercel DNS**:

   | Type | Name | Value |
   |---|---|---|
   | TXT | `resend._domainkey` | `p=...` (the value Resend gives you) |
   | MX | `send` | `feedback-smtp.us-east-1.amazonses.com`, priority 10 |
   | TXT | `send` | `v=spf1 include:amazonses.com ~all` |
   | CNAME | `links` | `links1.resend-dns.com` |

   **Do not proxy the CNAME.** Vercel DNS has no proxy toggle, so
   this is only a warning if you later move to Cloudflare.
3. Resend → **Restart verification** if it has not passed in 15 minutes.
4. Resend → **Segments → New** → name it `waitlist` → copy the uuid.
5. Resend → **API Keys → Create** → copy the key.

**In Vercel:**

```
RESEND_API_KEY=re_xxxxxxxxx
RESEND_SEGMENT_ID=<the segment uuid>
RESEND_FROM=Atrium <hello@atrium-rag.space>
```

Both the key and the segment are needed — the key alone is not enough,
because a contact has to land in the segment the broadcast goes to.

**After that, add DMARC.** Once SPF and DKIM pass:

| Type | Name | Value |
|---|---|---|
| TXT | `_dmarc` | `v=DMARC1; p=none; rua=mailto:dmarc@atrium-rag.space` |

Start at `p=none`. Move to `quarantine` only after a week of clean
passes. This is what stops someone spoofing your domain in a "re: your
Atrium waitlist" email.

**Resend's free tier:** 3,000 transactional emails a month, 100 a day
(UTC, resets at midnight), 1,000 contacts a month for broadcasts, 3
verified domains, 10 requests a second. For a waitlist that is
generous. Note the 100/day cap — if the site ever gets scraped, the
route's own 5-per-10-minutes-per-IP throttle is what protects you.

### 2.2 Receiving — an address you can read

This is a different problem, and it needs a different service: one
that *receives* mail and puts it in an inbox.

**The simplest option that needs no DNS change: forward to your Gmail.**

- **Forward Email** (~$3/year for a single domain alias) — add an MX
  record and a TXT SPF record in Vercel DNS, and mail to
  `hello@atrium-rag.space` lands in your Gmail. It is forwarding
  only: you can read and reply, but the reply comes from your Gmail.
  Perfect for a `hello@` address on a project page.
- **Cloudflare Email Routing** — free, and it works with *any* MX
  provider, but it needs Cloudflare to be the DNS host. Using it means
  moving the nameservers off Vercel, which would mean moving the
  domain's DNS off Vercel too (the records are all there, it is a
  copy-paste, but it is work). It also cannot send — replying from
  the alias is not supported.
- **Zoho Mail free tier** — a real mailbox with IMAP, but the free
  tier is limited to a few users on one domain and it wants its own
  MX records.
- **Google Workspace** — a real mailbox with the best mail client, but
  it starts around $7/user/month and there is no free tier for a
  custom domain.

**My recommendation: Forward Email, and point `hello@` at your Gmail.**
It is the smallest thing that gives a human-readable address on your own
domain, it needs only two DNS records, and it costs almost nothing. If
you later want a full mailbox, move to Zoho or Google Workspace then —
nothing else about the setup depends on which one you chose.

**The addresses to set up:**

| Address | Goes to | Purpose |
|---|---|---|
| `hello@` | your inbox | the address on the site and in the docs |
| `support@` | your inbox | issues and questions |
| `security@` | your inbox | the one `SECURITY.md` should point at |
| `noreply@` | Resend | the transactional sender, with SPF/DKIM on Resend |

**One thing to change in the library.** `atrium-rag/SECURITY.md` currently
says the maintainer's address is in the README, and the README does not
have one. Put `security@atrium-rag.space` in the README and point
`SECURITY.md` at it. A security policy that says "email the address in
the README" when the README has no address is a dead end for exactly
the person who would report a vulnerability.

---

## 3. The rest of the domain and IP setup

### 3.1 DNSSEC

```
whois atrium-rag.space → DNSSEC: unsigned
```

It is **not** signed. Vercel DNS does not offer DS records, so enabling
it means the registrar (Name.com) has to hold the DS record while
Vercel holds the zone — which is not possible, because a DNSSEC-signed
zone must serve the signatures from the same nameservers that are
authoritative for it.

**So: DNSSEC on this domain requires moving the DNS to a registrar
that does both** — Cloudflare Registrar, or Name.com's own DNS if it
offers the option. Neither is required for the launch. It is a "nice
to have, later" item, and moving DNS is more risk than the benefit
justifies right now. If you want it later, moving to Cloudflare
Registrar also unlocks Cloudflare Email Routing, which is why it is the
natural next step.

### 3.2 The other records you want

| What | Where | Why |
|---|---|---|
| `NEXT_PUBLIC_SITE_URL` | Vercel env | the canonical, sitemap, robots, JSON-LD |
| `RESEND_*` (3 vars) | Vercel env | the waitlist |
| `BUTTONDOWN_API_KEY` or `FORMSPREE_FORM_ID` | Vercel env | the fallback, if you keep one |
| DMARC TXT | Vercel DNS | anti-spoofing, after SPF/DKIM pass |
| Search Console property | Google | submit the sitemap |
| Bing Webmaster Tools | Microsoft | feeds Copilot and Edge answers |

### 3.3 Search Console, now that the domain is live

1. <https://search.google.com/search-console> → **Add property** →
   **Domain** → `atrium-rag.space`.
2. Verify with a **TXT record** in Vercel DNS (Google gives you the
   value).
3. **Sitemaps** → submit `sitemap.xml`. It is already generated —
   eight URLs, from the routes that exist.
4. Within a week, check **Performance → Search queries**. That is the
   list of what people are actually searching for, which tells you what
   to write next. It is the single most useful report on this whole
   list, because it replaces guessing.

### 3.4 What you have not got, and do not need yet

- **A `.com`.** `atrium-rag.space` is fine. `.space` is a cheap TLD
  with a slightly worse reputation in some corporate mail filters,
  which is why the transactional sender is `noreply@` on Resend rather
  than a `hello@` address — a confirmation from a `hello@` on an
  unfamiliar TLD is one that goes to spam.
- **An ICP filing or any other regional registration.** `.space` is
  gTLD; there is no country-code registry process involved.
- **A trademark.** `Atrium` is an existing name in real estate
  (Atrium Living, the apartment operator). That is not a problem for
  an open-source project using it descriptively, and you are not
  selling a service under the name yet. If Atrium Cloud ever becomes
  a hosted product with revenue, that is the point to look at it.

---

## 4. Should `atrium-web` be private?

**You can, and it will not break anything.** Vercel deploys from a
private repository on every plan.

**I would leave it public, for two reasons.**

1. It is a portfolio piece. A reader who lands on the site and wants
   to know how the page is built can read the source, and the site
   says "built for engineers who read the source". A private
   repository contradicts that sentence.
2. The `check-numbers` CI job — the one that guarantees the site
   cannot quote a number the library does not produce — is only
   meaningful if someone else can run it.

The **library** repo should stay public regardless: it is MIT, its
README says `pip install atrium-rag`, and every link in the docs and on
the site points at it. A private library repo with a public install
command is incoherent, and it is the one thing a person evaluating
your work will check first.

If you make `atrium-web` private anyway, do not make `atrium-rag`
private. They are not the same decision.

---

## 5. The four numbers to watch

| | where | what it tells you |
|---|---|---|
| PyPI downloads | <https://pypistats.org/packages/atrium-rag> | installs — the real measure of interest |
| Vercel Analytics | project → Analytics | visitors and **referrers** — which platform brought them |
| Resend contacts | Resend → Contacts / Segments | the waitlist, and the number you write "you asked for this" about |
| GitHub stars | the repo | interest without adoption |

Read them weekly. **Downloads rising while stars do not** means the
README is wrong. **Stars rising while downloads do not** means the
install path is broken — people like it and cannot get it running.

---

## 6. The order

1. `NEXT_PUBLIC_SITE_URL` → redeploy. **Ten minutes, and it fixes a
   live SEO bug.**
2. Resend: verify the domain, add the four DNS records, create the
   segment, set the three env vars, add DMARC.
3. Forward Email: two DNS records, `hello@` and `support@` into your
   inbox.
4. Put `security@atrium-rag.space` in the README and in `SECURITY.md`.
5. Search Console: verify with a TXT record, submit the sitemap.
6. Bing Webmaster Tools: two minutes.
7. Flip the apex/www arrangement if you want the apex to serve.
8. Redirect the old `vercel.app` URL.
9. Push `atrium-web` — **4 commits are unpushed**, and the deployment
   is currently running a build from your laptop.
