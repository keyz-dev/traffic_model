# Image Dataset Setup (Hair vs Other)

This adds a robust DuckDuckGo-based downloader and an optional Pinterest scraper to build a dataset that distinguishes hair-related items from non-hair items.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-images.txt
# Optional for Pinterest (downloads browser binaries):
python -m playwright install chromium
```

## Queries

Edit queries in `data/queries.yaml`.

## DuckDuckGo Downloader

Downloads images for each keyword into `dataset/train/<category>/`.

```bash
python scripts/ddg_image_download.py \
  --queries_file data/queries.yaml \
  --base_dir dataset/train \
  --limit 35 \
  --concurrency 16 \
  --min_size 256
```

Notes:
- Automatic validation (real images only), minimum size enforced
- Perceptual-hash de-duplication across a category
- Skips non-image URLs and tiny thumbnails

## Pinterest Scraper (Optional)

Searches Pinterest and downloads results for a single keyword.

```bash
# One-time install of browser binaries
python -m playwright install chromium

# Unauthenticated scrape
python scripts/pinterest_scraper.py "lace front wig product" --limit 120 --output_dir dataset/pinterest/hair

# Optional login (may increase result availability)
export PINTEREST_EMAIL="you@example.com"
export PINTEREST_PASSWORD="your_password"
python scripts/pinterest_scraper.py "hair conditioner bottle" --login --limit 120 --output_dir dataset/pinterest/hair
```

## Next Steps

1. Manually review and clean images in each folder.
2. Create `dataset/validation/{hair,other}` and move ~20% of cleaned images from `dataset/train` into validation.
3. Consider adding a small `dataset/test` split for final evaluation.