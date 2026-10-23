# Environment and execution

Declared upstream dependency files:
- `requirements.txt`

Current execution environment is recorded in the master package verification/ENVIRONMENT.json. `python run_checks.py` records and fails on missing dependencies; it does not install packages or suppress tests. The five dependency-blocked repositories require the missing original format/CAD libraries before their checks can be considered complete. A successful local check is not a production or public-release approval.
