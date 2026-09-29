# Replacing the routing bot

**To:** Ritu Deshpande, Head of D2C Operations · **Cc:** Farhan, Meenal, Tanmay · **Date:** 30 September 2026

## The decision

**Switch the bot off, and replace it with the new router after a two-week side-by-side check.** Judge the router on
whether requests reach the team that closes them, not on whether it matches the bot's old labels.

## Why the 90% target needs to change

The labels in the export show where the bot *sent* each request, which is not always where it belonged. On your
own records, the bot's choice was the team that finally closed the request **77 times in 100**. We built a copy of
the bot that matched its labels 92% of the time. It routed no better than the bot, because it had learned the bot's
mistakes. So we trained the router on where requests were actually closed.

## The number

On the three most recent months it had never seen (April–June 2026, 2,135 requests):

* **New router: 86 in 100** go straight to the team that closes them. **Bot: 77 in 100.**
* When the customer says what they need (84% of requests), the router gets **98 in 100** right.
* The rest say only "please call me about my purifier". Past requests like that were closed by all seven teams, so
  no system and no person can route them from the words alone. That is why 90% against real outcomes is out of
  reach for *any* router working from the message. **One question at intake** ("Is this about a fault, a
  delivery, installation, warranty, payment, spares or how to use it?") could lift the router to about 98%.
* Meenal's complaints are confirmed. Of 626 customers who said they had paid, the bot sent 525 to Billing, and
  Billing closed **one**. Purifier breakdowns were sent to Consumables.

## The rupees

| | per year |
|---|---|
| Bot licence no longer paid | **Rs 3.2 lakh** |
| Fewer transfers and repeat contacts (wrong first team falls from 23% to 14%) | **Rs 5.5 lakh** |
| **Total saving** | **about Rs 8.7 lakh** |
| If the intake question works (to be proven in a pilot) | up to Rs 16 lakh |

Misrouting costs about Rs 14.7 lakh a year today: 3,129 transfers at Rs 305, plus Rs 260 for each repeat contact.
Misrouted requests also take twice as long to close (17 hours against 8).
**For Farhan: the router costs Rs 0 per request.** It runs on Kestrel's own server in under 2 milliseconds, with
no AI subscription and no bill that grows with volume.

## For headcount: use real volumes, not the bot's

Requests closed per month over the last six months, with what the bot's labels suggest in brackets:
Repairs 166 (208) · Returns 107 (82) · Installs 105 (78) · Product Advice 91 (77) · Billing 90 (114) ·
Warranty 90 (68) · Filters & Consumables 73 (96). Planning on the bot's numbers would over-staff Repairs, Billing
and Consumables, and under-staff Installs, Returns and Warranty.

## What to do next week

1. **Agree the measure:** the share of requests closed by the first team they reach. Today it is 77%; target 86%+.
   Tanmay can report it weekly from the resolution log.
2. **Start a two-week side-by-side run (Tanmay):** the bot keeps routing while the router's choice is logged
   next to it. Time the switch-off for after this run, so the 86% is confirmed on live requests first.
3. **Pilot the intake question on one channel (Meenal)**, for example WhatsApp. The seven options are already built
   into the router's screen.
4. **Hold headcount decisions** until you have seen the real volumes above.

*Caveats: tested on history, not on live traffic. About 2% of requests in the log were closed by an unrelated team,
which looks like record-keeping noise. Wording customers have never used before is routed correctly about 87% of
the time, and the router flags these for a quick check.*
