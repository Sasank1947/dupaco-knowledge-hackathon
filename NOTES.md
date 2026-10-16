# Notes

Sasank Sandeep Padamata

My solution is in `submission/solution.py`. It needs no API keys and no extra packages, only standard Python 3. It also makes no network calls.

To run it:

```bash
python harness/run_eval.py --submission submission --eval eval/public_eval.jsonl
```

On the practice set I get 0.897 overall: accuracy 0.897, consistency 1.0, speed 1.0. Most answers come back in well under 0.1 seconds. Loading all five snapshots takes about 4 seconds.

## How it works

I keep every version of every page, along with the snapshot it came from. From that I build a view of the site as it looked on each snapshot date. A page that disappeared in a later crawl stays in the view, marked as no longer live, so I can still use it if nothing newer covers the topic. When a question comes in with an `as_of` date, I answer from the latest snapshot on or before that date. Without a date, I answer from the newest snapshot.

Each view has two indexes:

- **Rate tables.** I parse every markdown table into rows. For each table I keep its caption and the section it sits under, the column headers, the "Rates are effective as of" date printed under it, and the page it came from. I also tag the page type: rate sheet, product page, PDF, news, blog, landing page, calculator, or listing.
- **Text.** I split pages into chunks by heading and search them with BM25. Each chunk remembers its page type and, for blogs and press releases, the publish date.

When a question comes in, I first work out what kind it is: a link, branch hours, an address, a rate, a phone number, a minimum balance, or a general question. Each kind has its own handler.

## How I decide what's current

1. **The newest effective date wins.** The same rate table shows up in a lot of places: the `/rates` page, product pages, calculators, landing pages, an old home-buying guide. I trust the copy with the latest "effective as of" date. For example, the vehicle calculator still says 5.34% for a 60-month auto loan (effective 1/1/2026), while the auto loan page says 5.39% (effective 8/31/2026), so I answer 5.39%.
2. **Official pages beat everything else.** The rate sheet and product pages rank above PDFs, news, blogs, landing pages and calculators. Tag and listing pages barely count.
3. **Live pages beat pages that were taken down.** I still use removed pages for facts nobody updated, but a live page wins when both cover the same thing.
4. **For counts, the newest article wins.** The "About Dupaco" footer under old press releases says 20, 22, 23 or 24 branch locations depending on the year. I take the most recently published one, which says 22.
5. **Different tiers aren't conflicts.** The term, model year, balance or loan amount, and options like jumbo, Roth or Traditional, purchase or refinance, business, and intro are what pick the right row. A $250k mortgage goes to the smallest "$X or less" tier that fits it. Intro and promo rows only show up when someone asks about them.
6. **If the site doesn't say, I don't guess.** In these cases I return `confident=False` and put no numbers in the answer:
   - The product never has a published rate. Examples: construction loans, ARMs, FHA/VA loans, jumbo mortgages, business loans.
   - The term asked about isn't offered, or wasn't offered on the `as_of` date.
   - The branch's hours aren't on the site.
   - The question isn't covered anywhere.

   If a term used to exist and was dropped, like the 25-month certificate special, I say it's no longer listed and leave out the old rate.

## conflicts()

For every rate tier I build a key out of product, term, model-year tier, balance or loan tier, and the jumbo or intro flag. That way "2023-26 Model Year" and "2024-27 Model Year" still count as the same top tier. I collect every value that key ever had, across pages and across snapshots. If there are two or more, that's a conflict. The value I pick is the one from the freshest official page that is still live, and the reason lists where the old values came from. I also add two text conflicts: the branch count and how many times Forbes has recognized Dupaco. The ones still visible on the live site are listed first.

## Speed and consistency

All the parsing happens in `ingest()`, so answering a question is one table scan or one BM25 lookup. Answers are deterministic and cached, so asking the same thing twice gives the same text. Reworded questions go through the same logic and come out with the same numbers.

## What I'd improve

- Answers are sentences taken straight from the site, not rewritten summaries.
- A few facts are split across pages. For example, the join page says "live or work in our charter area", but the actual states are on other pages. Those answers can come back incomplete.
