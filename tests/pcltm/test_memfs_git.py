from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from pcltm.memfs_store import MemFSStore


GIT_AVAILABLE = shutil.which("git") is not None
needs_git = pytest.mark.skipif(not GIT_AVAILABLE, reason="git not available")


def write_memory_file(root: Path, relative_path: str = "pinned/item.md", body: str = "Body content.") -> Path:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\n"
        'description: "Git test memory"\n'
        "authority: pinned\n"
        "mode_scope: [work]\n"
        "buckets: [git]\n"
        "source: pcltm\n"
        'last_reviewed: "2026-05-23"\n'
        "---\n\n"
        f"{body}\n",
        encoding="utf-8",
    )
    return path


def git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def configure_git_identity(root: Path) -> None:
    git(root, "config", "user.email", "memfs-test@example.invalid")
    git(root, "config", "user.name", "MemFS Test")


@needs_git
def test_ensure_git_repo_creates_repo(tmp_path: Path) -> None:
    store = MemFSStore(tmp_path)

    assert store.ensure_git_repo() is True

    assert (tmp_path / ".git").is_dir()


@needs_git
def test_ensure_git_repo_idempotent(tmp_path: Path) -> None:
    store = MemFSStore(tmp_path)

    assert store.ensure_git_repo() is True
    assert store.ensure_git_repo() is True

    assert (tmp_path / ".git").is_dir()


@needs_git
def test_has_uncommitted_false_on_clean(tmp_path: Path) -> None:
    store = MemFSStore(tmp_path)
    assert store.ensure_git_repo() is True

    assert store.has_uncommitted_changes() is False


@needs_git
def test_has_uncommitted_true_after_file_change(tmp_path: Path) -> None:
    store = MemFSStore(tmp_path)
    assert store.ensure_git_repo() is True

    write_memory_file(tmp_path)

    assert store.has_uncommitted_changes() is True


@needs_git
def test_commit_changes_creates_commit(tmp_path: Path) -> None:
    store = MemFSStore(tmp_path)
    assert store.ensure_git_repo() is True
    configure_git_identity(tmp_path)
    write_memory_file(tmp_path)

    assert store.commit_changes("pinned: add test memory") is True

    result = git(tmp_path, "rev-list", "--count", "HEAD")
    assert result.stdout.strip() == "1"


@needs_git
def test_commit_changes_with_message(tmp_path: Path) -> None:
    store = MemFSStore(tmp_path)
    assert store.ensure_git_repo() is True
    configure_git_identity(tmp_path)
    write_memory_file(tmp_path)
    message = "episodic: capture task reason"

    assert store.commit_changes(message) is True

    result = git(tmp_path, "log", "-1", "--pretty=%B")
    assert result.stdout.strip() == message


@needs_git
def test_commit_fails_on_empty_no_changes(tmp_path: Path) -> None:
    store = MemFSStore(tmp_path)
    assert store.ensure_git_repo() is True
    configure_git_identity(tmp_path)

    assert store.commit_changes("pinned: no changes") is False


@needs_git
def test_runtime_reads_work_without_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_memory_file(tmp_path, body="Runtime reads are independent of git.")
    store = MemFSStore(tmp_path)
    monkeypatch.setattr(shutil, "which", lambda name: None if name == "git" else shutil.which(name))

    with pytest.warns(RuntimeWarning, match="git binary not available"):
        assert store.ensure_git_repo() is False
    frontmatter, body = store.read_file("pinned/item.md")

    assert frontmatter.description == "Git test memory"
    assert "independent of git" in body


def test_ensure_git_returns_false_on_broken_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = MemFSStore(tmp_path)
    monkeypatch.setattr(shutil, "which", lambda name: None if name == "git" else shutil.which(name))

    with pytest.warns(RuntimeWarning, match="git binary not available"):
        assert store.ensure_git_repo() is False
    assert not (tmp_path / ".git").exists()
