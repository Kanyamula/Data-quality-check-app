# African Tropical Plants — Geospatial Data Quality Validator

A Streamlit app that runs Darwin Core / BDQ (Biodiversity Data Quality) tests
against an uploaded CSV of specimen records — coordinate/country/state
consistency, coordinate ranges, geodetic datum, ISO country codes, and more.

## Quick start

### Windows (Command Prompt)

1. Install Python 3.9 or later from https://python.org if you don't already
   have it. During setup, check **"Add python.exe to PATH."**
2. Unzip this project anywhere, e.g. `C:\Users\you\Documents\dq-validator`.
3. Open Command Prompt, then run:

   ```
   cd C:\Users\you\Documents\dq-validator
   run_windows.bat
   ```

   (Or just double-click `run_windows.bat` in File Explorer.)

That's it — the first run creates a virtual environment and installs
dependencies (a few minutes), then opens the app in your browser. Every run
after that starts in seconds.

### macOS / Linux (Terminal)

1. Make sure Python 3.9+ is installed (`python3 --version`).
2. Unzip this project, then:

   ```
   cd /path/to/dq-validator
   chmod +x run_mac_linux.sh
   ./run_mac_linux.sh
   ```

## Manual setup (any OS)

If you'd rather run the commands yourself instead of using the scripts
above:

```
python -m venv .venv

# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Streamlit opens the app automatically at `http://localhost:8501`. To stop
it, press `Ctrl+C` in the terminal/Command Prompt window.

## Using the app

1. Open the app in your browser (it opens automatically; if not, go to the
   URL shown in the terminal).
2. Upload a CSV with Darwin Core-style columns (`decimalLatitude`,
   `decimalLongitude`, `country`, `countryCode`, `stateProvince`,
   `coordinateUncertaintyInMeters`, `geodeticDatum`,
   `minimumElevationInMeters`, `maximumElevationInMeters`).
3. The app runs all BDQ tests and shows results, with a sidebar reporting
   the live status of each source authority (TGN, EPSG, Natural Earth,
   Marine Regions EEZ, ISO 3166 country codes).

The first time you run the coordinate/country-boundary tests, the app
downloads real geospatial reference data (Natural Earth admin-1 boundaries
and Marine Regions EEZ) — this can take a minute or two, but it's cached to
disk afterward so subsequent runs are fast.

## Optional: GitHub-backed upload history

By default, each session's upload history lives only in memory for that
session — this is entirely optional and the app works fully without it. If
you want upload history to persist across restarts (saved to a GitHub
repo):

1. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`.
2. Fill in `GITHUB_TOKEN` (a GitHub personal access token with repo/contents
   write access — create one at https://github.com/settings/tokens) and
   `GITHUB_REPO` (e.g. `your-username/your-repo`).
3. Restart the app.

**Never commit `.streamlit/secrets.toml`** — it's already listed in
`.gitignore` for exactly this reason, since it holds a real credential.

## Putting this project on GitHub

This folder is already a git repository (`git log` will show the initial
commit). To push it to your own GitHub account:

1. Create a new, empty repository on GitHub (don't initialize it with a
   README — this project already has one).
2. In this project folder, run:

   ```
   git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git
   git branch -M main
   git push -u origin main
   ```

3. Anyone can now get a runnable copy of the app with:

   ```
   git clone https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git
   cd YOUR-REPO-NAME
   ```

   and then follow the "Quick start" steps above.

## Troubleshooting

- **`geopandas` fails to install on Windows:** this is rare with modern
  versions (this project pins `geopandas>=0.13`, which ships prebuilt
  wheels for Windows/macOS/Linux), but if `pip install -r requirements.txt`
  fails specifically on `geopandas`, try installing
  [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and running
  `conda install -c conda-forge geopandas` before re-running
  `pip install -r requirements.txt`.
- **"python is not recognized..." on Windows:** Python isn't on your PATH.
  Reinstall Python from https://python.org and check "Add python.exe to
  PATH", or use the Python launcher: `py -m venv .venv`.
- **Sidebar shows a source authority as "Not available":** that data source
  (TGN, EPSG, etc.) is temporarily unreachable — the affected tests will
  report `EXTERNAL_PREREQUISITES_NOT_MET` for that run rather than silently
  giving wrong results. This is expected behavior, not a bug.
