# Archived tests

For this public-ready push the repository maintainers requested tests and benchmarking artifacts be removed from the tracked tree to keep the public package lightweight.

What was done
- The `tests/` and `testing_benchmarking/` directories were removed from the repository tree.

How to restore locally (if you need the tests again)
1. If you still have the original files locally, restore them to the repository root.
2. Or, retrieve them from the previous commit before this cleanup. For example:

```bash
# show commits
git log --oneline

# find the commit hash prior to this cleanup and restore specific files
git checkout <commit-hash> -- tests/ testing_benchmarking/
```

Security note
- Tests may reference internal or example API keys and example data. Do not reintroduce secrets into the public repository.
