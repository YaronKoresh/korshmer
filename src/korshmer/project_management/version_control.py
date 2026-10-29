import logging
import re
import subprocess
from pathlib import Path
from typing import Any


class VersionControlManager:
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.repo_dir = self.base_dir / "saved_data"
        self.enabled = False
        self._check_git_availability()

    def _check_git_availability(self):
        try:
            result = subprocess.run(
                ["git", "--version"],
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=5,
            )
            if result.returncode == 0:
                logging.info(f"Git available: {result.stdout.strip()}")
                if (self.repo_dir / ".git").exists():
                    self.enabled = True
                    logging.info("Git repository detected")
            else:
                logging.warning("Git command failed")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            logging.warning(
                "Git not found. Version control features will be limited."
            )
        except Exception as e:
            logging.error(f"Error checking Git availability: {e}")

    def _validate_branch_name(self, branch_name: str) -> bool:
        if not branch_name or not isinstance(branch_name, str):
            return False
        if branch_name.startswith(".") or branch_name.startswith("-"):
            return False
        if ".." in branch_name:
            return False
        invalid_chars = {"~", "^", ":", "?", "*", "[", " ", "\\"}
        if any(c in branch_name for c in invalid_chars):
            return False
        if branch_name.endswith(".lock") or branch_name.endswith("/"):
            return False
        return True

    def _validate_remote_name(self, name: str) -> bool:
        if not name or not isinstance(name, str):
            return False
        return bool(re.match(r"^[a-zA-Z0-9_][a-zA-Z0-9_-]*$", name))

    def _validate_remote_url(self, url: str) -> bool:
        if not url or not isinstance(url, str):
            return False
        allowed_protocols = ["https://", "git://", "ssh://", "git@"]
        if not any(url.startswith(proto) for proto in allowed_protocols):
            return False
        return True

    def _validate_remote_and_branch(self, remote: str, branch: str) -> bool:
        if not self._validate_remote_name(remote):
            logging.error(f"Invalid remote name: {remote}")
            return False
        if not self._validate_branch_name(branch):
            logging.error(f"Invalid branch name: {branch}")
            return False
        return True

    def _validate_commit_message(self, message: str) -> bool:
        if not message or not isinstance(message, str):
            return False
        if len(message.strip()) == 0:
            return False
        if len(message) > 5000:
            return False
        if any(ord(c) < 32 and c not in "\n\r\t" for c in message):
            return False
        return True

    def _run_git_command(
        self, args: list[str], cwd: Path | None = None
    ) -> tuple[bool, str]:
        try:
            if cwd is None:
                cwd = self.repo_dir

            result = subprocess.run(
                ["git"] + args,
                cwd=str(cwd),
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
            )

            if result.returncode == 0:
                return True, result.stdout
            else:
                logging.warning(
                    f"Git command failed: {' '.join(args)}\n{result.stderr}"
                )
                return False, result.stderr

        except subprocess.TimeoutExpired:
            logging.error("Git command timed out")
            return False, "Command timed out"
        except Exception as e:
            logging.error(f"Git command error: {e}")
            return False, str(e)

    def init_repository(self) -> bool:
        if (self.repo_dir / ".git").exists():
            logging.info("Git repository already exists")
            self.enabled = True
            return True

        success, output = self._run_git_command(["init"])
        if success:
            self.enabled = True

            gitignore_path = self.repo_dir / ".gitignore"
            if not gitignore_path.exists():
                with open(gitignore_path, "w") as f:
                    f.write(
                        "*.wav\n*.mp3\n*.flac\n*.ogg\n*.aac\n*.aiff\n*.aif\n"
                    )
                    f.write("temp/\n__pycache__/\n*.pyc\n.DS_Store\n")
                    f.write("models/\nlogs/\n")

                added, _ = self._run_git_command(["add", ".gitignore"])
                if added:
                    committed, _ = self._run_git_command(
                        ["commit", "-m", "Add .gitignore"]
                    )
                    if not committed:
                        logging.warning("Failed to commit .gitignore")
                else:
                    logging.warning("Failed to add .gitignore to git index")

            logging.info("Git repository initialized")
            return True

        return False

    def commit(self, message: str, files: list[Path] | None = None) -> bool:
        if not self.enabled:
            return False

        if not self._validate_commit_message(message):
            logging.error("Invalid commit message")
            return False

        if files:
            for file_path in files:
                success, _ = self._run_git_command(
                    ["add", "--", str(file_path)]
                )
                if not success:
                    return False
        else:
            success, _ = self._run_git_command(["add", "."])
            if not success:
                return False

        success, output = self._run_git_command(["commit", "-m", message])
        return success

    def get_commit_history(self, max_count: int = 10) -> list[dict[str, Any]]:
        if not self.enabled:
            return []

        success, output = self._run_git_command(
            [
                "log",
                f"--max-count={max_count}",
                "--pretty=format:%H|%an|%ae|%ad|%s",
                "--date=iso",
            ]
        )

        if not success:
            return []

        commits = []
        for line in output.strip().split("\n"):
            if not line:
                continue
            parts = line.split("|", 4)
            if len(parts) == 5:
                commits.append(
                    {
                        "hash": parts[0],
                        "author": parts[1],
                        "email": parts[2],
                        "date": parts[3],
                        "message": parts[4],
                    }
                )

        return commits

    def create_branch(self, branch_name: str) -> bool:
        if not self.enabled:
            return False

        if not self._validate_branch_name(branch_name):
            logging.error(f"Invalid branch name: {branch_name}")
            return False

        success, _ = self._run_git_command(["branch", branch_name])
        return success

    def checkout_branch(self, branch_name: str) -> bool:
        if not self.enabled:
            return False

        if not self._validate_branch_name(branch_name):
            logging.error(f"Invalid branch name: {branch_name}")
            return False

        success, _ = self._run_git_command(["checkout", branch_name])
        return success

    def merge_branch(self, branch_name: str) -> bool:
        if not self.enabled:
            return False

        if not self._validate_branch_name(branch_name):
            logging.error(f"Invalid branch name: {branch_name}")
            return False

        success, _ = self._run_git_command(["merge", branch_name])
        return success

    def get_branches(self) -> list[str]:
        if not self.enabled:
            return []

        success, output = self._run_git_command(["branch", "--list"])
        if not success:
            return []

        branches = []
        for line in output.strip().split("\n"):
            branch = line.strip()
            if branch.startswith("* "):
                branch = branch[2:]
            if branch:
                branches.append(branch)

        return branches

    def tag_version(self, tag_name: str, message: str | None = None) -> bool:
        if not self.enabled:
            return False

        if not self._validate_branch_name(tag_name):
            logging.error(f"Invalid tag name: {tag_name}")
            return False

        if message is not None and not self._validate_commit_message(message):
            logging.error("Invalid tag message")
            return False

        args = ["tag"]
        if message:
            args.extend(["-a", tag_name, "-m", message])
        else:
            args.append(tag_name)

        success, _ = self._run_git_command(args)
        return success

    def get_diff(self, file_path: Path | None = None) -> str:
        if not self.enabled:
            return ""

        args = ["diff"]
        if file_path:
            args.extend(["--", str(file_path)])

        success, output = self._run_git_command(args)
        return output if success else ""

    def revert_file(self, file_path: Path) -> bool:
        if not self.enabled:
            return False

        success, _ = self._run_git_command(["checkout", "--", str(file_path)])
        return success

    def stash_changes(self, message: str | None = None) -> bool:
        if not self.enabled:
            return False

        args = ["stash", "push"]
        if message:
            args.extend(["-m", message])

        success, _ = self._run_git_command(args)
        return success

    def stash_pop(self) -> bool:
        if not self.enabled:
            return False

        success, _ = self._run_git_command(["stash", "pop"])
        return success

    def push_to_remote(
        self, remote: str = "origin", branch: str = "main"
    ) -> bool:
        if not self.enabled:
            return False

        if not self._validate_remote_and_branch(remote, branch):
            return False

        success, _ = self._run_git_command(["push", remote, branch])
        return success

    def pull_from_remote(
        self, remote: str = "origin", branch: str = "main"
    ) -> bool:
        if not self.enabled:
            return False

        if not self._validate_remote_and_branch(remote, branch):
            return False

        success, _ = self._run_git_command(["pull", remote, branch])
        return success

    def add_remote(self, name: str, url: str) -> bool:
        if not self.enabled:
            return False

        if not self._validate_remote_name(name):
            logging.error(f"Invalid remote name: {name}")
            return False

        if not self._validate_remote_url(url):
            logging.error(f"Invalid remote URL: {url}")
            return False

        success, _ = self._run_git_command(["remote", "add", name, url])
        return success

    def get_status(self) -> dict[str, Any]:
        if not self.enabled:
            return {
                "enabled": False,
                "repository_exists": False,
                "current_branch": None,
                "modified_files": [],
                "staged_files": [],
                "untracked_files": [],
            }

        success, output = self._run_git_command(["status", "--porcelain", "-z"])

        modified_files = []
        staged_files = []
        untracked_files = []

        if success:
            entries = output.split("\0")
            index = 0
            while index < len(entries):
                entry = entries[index]
                index += 1
                if not entry:
                    continue
                if len(entry) < 4:
                    logging.warning(
                        "Ignoring malformed git status line: %r", entry
                    )
                    continue
                status = entry[:2]
                filename = entry[3:]
                if "R" in status or "C" in status:
                    index += 1

                if status[0] in ["M", "A", "D", "R", "C"]:
                    staged_files.append(filename)
                if status[1] in ["M", "D"]:
                    modified_files.append(filename)
                if status == "??":
                    untracked_files.append(filename)

        success, branch_output = self._run_git_command(
            ["branch", "--show-current"]
        )
        current_branch = branch_output.strip() if success else None

        return {
            "enabled": self.enabled,
            "repository_exists": (self.repo_dir / ".git").exists(),
            "current_branch": current_branch,
            "modified_files": modified_files,
            "staged_files": staged_files,
            "untracked_files": untracked_files,
        }
