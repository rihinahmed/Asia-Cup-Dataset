# AsiaCupT20 dataset paper (IEEE conference format, LaTeX)

This folder holds the full paper as LaTeX source, one file for each section. The paper is **6 pages** in the official `IEEEtran` conference format, which is the maximum allowed. `main.pdf` is a compiled copy so you can see the layout.

## 1. What the paper is about

It describes a dataset of all 32 matches of the Asia Cup T20 tournaments of 2022 and 2025 (both played in the UAE). Each match has 58 fields, from the toss and scores to ground size and short notes on pitch and weather. A second table has player and innings detail for 2025. The last part of the paper shows a small example analysis, to prove the data can be used.

## 2. Folder map

```
main.tex                    compile this file
references.bib              all references (BibTeX, IEEE style)
main.pdf                    compiled preview of the paper
sections/                   abstract, introduction, related_work, data_collection,
                            dataset_description, technical_validation,
                            baseline_analysis, limitations, conclusion, references
tables/                     glossary, summary, fields, checks, models
figures/                    the 7 figures used in the paper
data/                       asia_cup_complete_dataset.xlsx (the dataset itself)
code/analysis.py            reproduces every number and 5 of the figures
code/fig_extra.py           draws the pipeline and completeness figures
code/results.json           the numbers quoted in the paper
```

## 3. How to compile

Upload the folder to Overleaf, or use a TeX installation that has the `IEEEtran` class. Run pdfLaTeX twice. The result should be 6 pages. The reference list is written out in sections/references.tex, so no BibTeX step is needed (references.bib is kept as a backup source).

## 4. What was done

1. **Built the dataset.** The 2022 sheet and the 2025 file were cleaned and merged into one master table (32 matches, 58 columns).
2. **Fixed errors.** Nine swapped dates in 2022, one wrong toss winner (India v Pakistan, 28 Aug 2022), the team order in the 2025 file (Team 1 now always batted first), and the 2025 start times.
3. **Added fields.** Match ID, series, stage, city, 15 formula columns, and a source note for every pitch and weather entry.
4. **Researched 2025 pitch and weather** match by match. Nothing was guessed, so some cells are empty on purpose.
5. **Ran 13 automatic quality checks.** All passed (Table IV in the paper).
6. **Ran an example analysis** (Section VI).

## 5. Models that were run

Everything was run on the 31 matches decided in normal play. The one Super Over match is left out of win and loss models.

| What | Method | Why |
|---|---|---|
| Chasing and toss effects | Exact binomial test, Wilson ranges, Fisher exact test | Standard tools for small counts |
| Team strength | Elo rating updated match by match | Simple and easy to explain |
| Ground effects | Bayesian hierarchical logistic regression (4 chains of 30,000 steps) | Works with small samples and gives a range |
| Predicting the winner | Logistic regression, random forest, gradient boosting | Three common methods compared fairly |
| Baselines | Always pick the chasing side, pick the higher rated team | A model must beat these to be useful |
| Honest testing | 20 times repeated 5 fold cross validation, train on one year and test on the other, 300 shuffle permutation test | Every match is predicted by a model that has not seen it |

Two moments were tested: right after the toss, and at the break between innings. No settings were tuned.

**Main results (small sample, none decisive).** Chasing sides won 19 of 31 matches (61.3%, range 44 to 76%). Toss winners won 19 of 32 (59.4%). Logistic regression picked about 71% of winners against 61% for the simple baseline, with a permutation p value of 0.053.

To rerun the analysis, copy the workbook from `data/` next to `code/analysis.py`, name it `data.xlsx`, and run `python analysis.py` (needs pandas, numpy, scipy, scikit-learn, matplotlib).

## 6. Data source and licence (please read)

The 2025 part of the dataset starts from the public dataset by **K. Raza and F. Ali** (Zenodo, DOI 10.5281/zenodo.17228056, arXiv:2512.19740), which uses the **CC BY 4.0** licence. That licence allows reuse and changes, but only with credit and a clear statement of what was changed. The paper does both (Sections I, II and III, and the acknowledgment). This means the paper describes an **extended and corrected compilation**, not a dataset collected entirely from scratch, and the wording in the paper already says so. Please keep that wording.

## 7. Fill in or confirm before submitting

- Author names, affiliations and emails in `main.tex`; the acknowledgment (end of `sections/conclusion.tex`); the repository link (`sections/limitations.tex`).
- `data_collection.tex` says the 2022 rows were compiled by the authors. Name the websites or scorecards you used if you want to be specific.
- Check the target conference's own template and rules. Some ask for an anonymous first page or their own class file.
- Dashes: the writing has none. The only exceptions are page ranges in the reference list, which IEEE style requires, and two proper names (scikit-learn and the column name Wicket-to-Wicket Length).
