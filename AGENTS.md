# Experimentation Uplift Modeling

A reproducible Python study of randomized experiments, uplift learners, and treatment policies using the Criteo uplift dataset.

## Build commands

```bash
# Planned after Milestone 1; not runnable yet.
uv sync
uv run pytest
uv run ruff check .
uv run mypy src
```

## Platform support

- macOS and Linux; Python 3.12.
- Laptop development may use a fixed subset; this project has no deployment target.

## Key conventions

- Use only pre-treatment features f0 through f11; exposure and visit must not become model inputs.
- Freeze the analysis plan and test partition before final model comparison; use known treatment propensity in policy evaluation.
- Report estimated conditional effects as estimates, never as observed individual counterfactuals.

## Rules

- Default branch is `main`; never force-push it or rewrite shared history.
- Ask before adding any dependency or package manager.
- Never fabricate data, benchmarks, logs or results.
- Done means the relevant tests actually ran and passed; never skip or mock them to finish.
- No emojis anywhere: code, comments, commits, READMEs, docs.
- No em dashes (the long dash, U+2014) anywhere: code, comments, commits, READMEs, docs. Use a comma, colon, parentheses, or a new sentence instead.
- Keep README in sync with code changes, following its existing template.
- Comments are brief and explain why, only when the code doesn't already say it.
- Keep dataset splits and eval leakage controls intact.

## Commits

- Conventional Commits: `type(scope): summary`. Types: `feat`, `fix`, `refactor`, `perf`, `test`, `docs`, `build`, `ci`, `chore`; scope is optional.
- Summary in imperative mood, lowercase, no trailing period, at most 72 characters.
- One logical change per commit; keep formatting-only changes separate.
- Add a body only when the reason is not obvious: explain why, not what, wrapped at 72.
- Breaking changes: `type!:` in the summary plus a `BREAKING CHANGE:` footer.
- Commit only when asked. Never commit secrets, build output, or failing code.
