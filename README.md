# from app reviews to themes

Material for the talk: After Text Classification: 
Following Hikers Through an App Acquisition

Robert Held
29. October 2026



Open `reviews_to_themes.ipynb` and run it top to bottom. It goes:
reviews → taxonomy → training data → classifier → theme shares → dashboard.

## what's here

| | |
|---|---|
| `slides/` | the talk deck (PDF) |
| `reviews_to_themes.ipynb` | the whole pipeline, one notebook |
| `data/reviews_sample.csv` | 10,000 German reviews, komoot + 9 competitors, 2022 onwards. Ids replaced, emails, links and @names removed |
| `data/taxonomy.csv` | the codebook: themes, subthemes, definitions, German examples |
| `data/training_data.csv` | 4,301 labelled reviews the classifier learns from |
| `data/gold_heldout.csv` | 198 hand-labelled reviews, only for scoring |
| `models/` | the classifier used in the talk, if you'd rather not retrain |
| `dashboard/` | open `index.html` in a browser. Built from all German reviews; the notebook can rebuild it from the sample |

## what's not here

No scrapers and no full review text. Reviews belong to the people who wrote them and to the stores' terms, so this is a sample.

## setup

```
pip install pandas scikit-learn joblib matplotlib
```

German only. The classifier was trained on German text and won't work on other languages.

Reach out to run in English

RobertFHeld@gmail.com

Rob Held · [robertforrestheld.com](https://www.robertforrestheld.com)
