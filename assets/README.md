# MATVEC vehicle reference photos

This directory holds locally-cached vehicle photos shown in the Streamlit
sidebar when a preset is selected. Files are downloaded by
`download_assets.py` from Wikimedia Commons via the
`Special:FilePath` redirect.

**Run once after cloning:**
```
python download_assets.py
```

**Re-download all:**
```
python download_assets.py --force
```

If a local file is missing, `app.py` falls back to the live Wikimedia URL
(or silently skips the image entirely on network failure). The script and
the fallback chain make image rendering offline-robust without making
network availability a hard requirement.

The `.jpg` files in this directory are runtime artifacts and may be
gitignored at the user's discretion.
