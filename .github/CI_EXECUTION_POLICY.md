# CI execution policy — owner directive, 2026-10-13

**Use available Free/included CI credits for necessary validation.** Being conservative means avoiding redundant jobs, not foregoing meaningful acceptance, security, browser, integration, or regression testing. Delaying work solely to preserve existing credits is not acceptable.

1. Check *this repository's* provider, own account budget, current free/included balance and configured trigger rules. Never borrow or assume another project/organization's 30,000-credit Free allocation.
2. Run necessary tests at meaningful checkpoints. Batch compatible checks, prefer efficient runners and caching; do not repeatedly test identical broken SHA without a correction.
3. Necessary testing **within verified existing free/included credit balances** is authorized. Paid plan upgrades, extra billing, new cards, debt/overages or accessing credentials are not implicitly authorized; raise those explicitly.
4. If a missing connector or CI trigger blocks automation, report the exact fixable provider setting or integration immediately, and continue other work rather than declaring an immutable blocker.
5. Enforce original repo safeguards: branch protection, reviews, private source, user data and production permissions. Do not claim unexecuted tests passed.
6. Record the exact tested commit, workflow URL, run results, consumed credits when accessible, and next work. Do not enable an all-push paid job accidentally.
7. This 2026-10-13 user override supersedes older blanket aversion to running hosted CI; it does **not** mandate adding CI where none is installed, and does not copy another repo's secrets, triggers, or budgets.

Applies across AI agents/chat sessions.
