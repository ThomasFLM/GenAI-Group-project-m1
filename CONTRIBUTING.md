# Contributing guide

This guide explains how to work together on the repo without getting in each other's way.

## Golden rule

**Never push directly to `main`.** Every change goes through a branch and a pull request reviewed by at least one other member.

## Getting started

```bash
git clone <repo-url>
cd <repo-name>
git checkout main
git pull
```

## Working on a task

1. Update `main`:
   ```bash
   git checkout main
   git pull
   ```
2. Create a branch for your task:
   ```bash
   git checkout -b feature/task-name
   ```
   Examples: `feature/document-chunking`, `fix/encoding-error`, `docs/readme`.
3. Make small, clear commits:
   ```bash
   git add .
   git commit -m "Add document chunking"
   ```
4. Push your branch:
   ```bash
   git push -u origin feature/task-name
   ```
5. On GitHub, open a **pull request** to `main`, describe what you did and ask for a review.
6. Once approved, merge it, then delete the branch.

## Good practices

- One branch = one task. Avoid branches that live for weeks.
- Run `git pull` on `main` before starting and regularly while working.
- Never commit API keys, passwords or `.env` files.
- Do not commit large data files: put them in `data/raw/` (ignored by git) and share them through the group drive.
- Write commit messages that say what changes, not "update" or "test".

## Merge conflicts

If Git reports a conflict when merging `main` into your branch:

```bash
git checkout feature/my-branch
git merge main
# open the conflicting files, keep the right content, remove the <<<<<<< ======= >>>>>>> markers
git add .
git commit
```

When in doubt, ask the group before forcing anything (avoid `git push --force`).

## Splitting the work

Use the GitHub **Issues** tab: one issue per task, assigned to one person. That way everyone can see who does what.
