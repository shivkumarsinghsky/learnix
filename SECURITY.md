# Security Policy

`learnix` is a learning repository maintained by [Shiv Kumar](https://github.com/shivkumarsinghsky).
It is not operated as a hosted service.

## Reporting a vulnerability

Please report security issues privately through GitHub's
[private vulnerability reporting](https://github.com/shivkumarsinghsky/learnix/security/advisories/new)
rather than opening a public issue.

## Model files

`learnix` saves models as plain JSON and never loads pickle files, because unpickling can
execute arbitrary code (see [ADR-002](docs/decisions/ADR-002-json-model-files-instead-of-pickle.md)).
The scikit-learn notebook uses joblib, which is pickle-based: only load joblib files you created
yourself.

## Secrets

The repository contains no credentials and the code reads none.
