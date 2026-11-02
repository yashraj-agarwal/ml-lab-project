# Delivery quality control

Prepared 2 November 2026 from the implemented CodeTrace AI project.

- Backend: 8 unit tests passed; actual HTTP verification passed all 11 checks, including validation, CORS and saved-model outputs.
- Frontend: TypeScript/Vite production build passed; full npm audit reported zero vulnerabilities. Browser checks covered predictions, metrics, project information and readable validation errors. Node.js 22 or newer is required.
- Cross-artifact verification: 45 checks passed; see Consistency_Check.json for individual results.
- Report: 8 pages, created from the downloaded IEEE Word template, with an academic cover and evidence appendix. All pages visually reviewed.
- Handbook: 23 pages, 12 chapters and 99 viva questions. All pages visually reviewed; mathematical worked examples independently checked.
- Team script: 15 pages, 18 slide speeches, balanced six-slide allocations, 90 distinct additional preparation questions and member-specific answers. All pages visually reviewed.
- Revision sheets: three one-page PDF sheets, visually reviewed.
- Presentation: 18 editable slides with 18 speaker notes. Artifact-tool finalization passed. Native PowerPoint opened and exported all slides; all 18 slides visually reviewed. Word opened/exported the report, handbook and script. PDF rendering used native Word export and Poppler because bundled LibreOffice was unavailable.
- Charts and experiment CSVs come from actual saved-model predictions or separately fitted experimental models. Experimental settings do not overwrite the deployed model.
- Verification backend and frontend servers were stopped; no listeners remained on ports 8000 or 5173.

## Explicit unresolved information

The faculty guide name remains a marked placeholder. Four publisher-verified journal articles are documented; the requested 20–25 qualifying indexed-paper quota was not verified, and indexing/access gaps are disclosed. Perfect scores refer only to the small synthetic dataset; real-world authorship detection performance is unvalidated.

## Archive contents

The delivery archive contains the evaluation_package directory, excluding private build files, temporary chart workspaces and Python caches. File_Manifest.json records SHA-256 hashes of all delivered files except itself. Reproduction scripts expect the accompanying project repository and its trained artifacts, as described in README.md.
