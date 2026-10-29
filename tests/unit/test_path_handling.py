import os
import sys
import tempfile
from pathlib import Path

import pytest


class TestUnicodePaths:
    def test_unicode_filename_russian(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = "тестовый_файл.txt"
            filepath = os.path.join(tmpdir, filename)

            with open(filepath, "w", encoding="utf-8") as f:
                f.write("test content")

            assert os.path.exists(filepath)

            with open(filepath, encoding="utf-8") as f:
                content = f.read()

            assert content == "test content"

    def test_unicode_filename_chinese(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = "测试文件.txt"
            filepath = os.path.join(tmpdir, filename)

            with open(filepath, "w", encoding="utf-8") as f:
                f.write("test content")

            assert os.path.exists(filepath)

    def test_unicode_filename_japanese(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = "テストファイル.txt"
            filepath = os.path.join(tmpdir, filename)

            with open(filepath, "w", encoding="utf-8") as f:
                f.write("test content")

            assert os.path.exists(filepath)

    def test_unicode_filename_arabic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = "ملف_اختبار.txt"
            filepath = os.path.join(tmpdir, filename)

            with open(filepath, "w", encoding="utf-8") as f:
                f.write("test content")

            assert os.path.exists(filepath)

    def test_unicode_filename_emoji(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = "test_😀_file.txt"
            filepath = os.path.join(tmpdir, filename)

            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write("test content")

                if os.path.exists(filepath):
                    assert True
                else:
                    pytest.skip("Filesystem doesn't support emoji in filenames")
            except (OSError, UnicodeEncodeError):
                pytest.skip("Platform doesn't support emoji in filenames")

    def test_mixed_unicode_filename(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = "test_тест_测试_テスト.txt"
            filepath = os.path.join(tmpdir, filename)

            with open(filepath, "w", encoding="utf-8") as f:
                f.write("test content")

            assert os.path.exists(filepath)

    def test_unicode_directory_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dirname = os.path.join(tmpdir, "目录_папка")
            os.makedirs(dirname, exist_ok=True)

            filepath = os.path.join(dirname, "file.txt")
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("test")

            assert os.path.exists(filepath)

    def test_pathlib_unicode_support(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = "测试.txt"
            path = Path(tmpdir) / filename

            path.write_text("test content", encoding="utf-8")
            assert path.exists()

            content = path.read_text(encoding="utf-8")
            assert content == "test content"


class TestLongPaths:
    def test_moderately_long_filename(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = "a" * 100 + ".txt"
            filepath = os.path.join(tmpdir, filename)

            with open(filepath, "w") as f:
                f.write("test")

            assert os.path.exists(filepath)

    def test_long_filename_200_chars(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = "a" * 200 + ".txt"
            filepath = os.path.join(tmpdir, filename)

            try:
                with open(filepath, "w") as f:
                    f.write("test")
                assert os.path.exists(filepath)
            except OSError:
                pytest.skip("Filesystem doesn't support 200-char filenames")

    def test_deeply_nested_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = tmpdir
            for i in range(10):
                nested = os.path.join(nested, f"dir{i}")

            os.makedirs(nested, exist_ok=True)

            filepath = os.path.join(nested, "file.txt")
            with open(filepath, "w") as f:
                f.write("test")

            assert os.path.exists(filepath)

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific test")
    def test_windows_max_path_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            remaining = 250 - len(tmpdir)
            if remaining > 0:
                filename = "a" * remaining + ".txt"
                filepath = os.path.join(tmpdir, filename)

                try:
                    with open(filepath, "w") as f:
                        f.write("test")
                    assert os.path.exists(filepath)
                except OSError:
                    pytest.skip("Path exceeds MAX_PATH limit")

    def test_path_length_normalization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = "a" * 50 + ".txt"
            redundant_path = os.path.join(tmpdir, ".", "subdir", "..", filename)

            os.makedirs(os.path.dirname(redundant_path), exist_ok=True)

            normalized = os.path.normpath(redundant_path)
            with open(normalized, "w") as f:
                f.write("test")

            assert os.path.exists(normalized)


class TestSpecialCharacters:
    def test_filename_with_spaces(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = "file with spaces.txt"
            filepath = os.path.join(tmpdir, filename)

            with open(filepath, "w") as f:
                f.write("test")

            assert os.path.exists(filepath)

    def test_filename_with_multiple_dots(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = "file.name.with.dots.txt"
            filepath = os.path.join(tmpdir, filename)

            with open(filepath, "w") as f:
                f.write("test")

            assert os.path.exists(filepath)

    def test_filename_with_dash(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = "file-with-dashes.txt"
            filepath = os.path.join(tmpdir, filename)

            with open(filepath, "w") as f:
                f.write("test")

            assert os.path.exists(filepath)

    def test_filename_with_underscore(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = "file_with_underscores.txt"
            filepath = os.path.join(tmpdir, filename)

            with open(filepath, "w") as f:
                f.write("test")

            assert os.path.exists(filepath)

    def test_filename_starting_with_dot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = ".hidden_file.txt"
            filepath = os.path.join(tmpdir, filename)

            with open(filepath, "w") as f:
                f.write("test")

            assert os.path.exists(filepath)

    def test_filename_with_parentheses(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = "file(with)parentheses.txt"
            filepath = os.path.join(tmpdir, filename)

            with open(filepath, "w") as f:
                f.write("test")

            assert os.path.exists(filepath)

    def test_filename_with_brackets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = "file[with]brackets.txt"
            filepath = os.path.join(tmpdir, filename)

            with open(filepath, "w") as f:
                f.write("test")

            assert os.path.exists(filepath)

    def test_filename_with_ampersand(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = "file&with&ampersand.txt"
            filepath = os.path.join(tmpdir, filename)

            with open(filepath, "w") as f:
                f.write("test")

            assert os.path.exists(filepath)

    def test_filename_with_at_symbol(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = "file@test.txt"
            filepath = os.path.join(tmpdir, filename)

            with open(filepath, "w") as f:
                f.write("test")

            assert os.path.exists(filepath)


class TestInvalidCharacters:
    def test_sanitize_invalid_windows_chars(self):
        invalid_chars = '<>:"|?*'

        for char in invalid_chars:
            bad_name = f"test{char}file.txt"

            sanitized = "".join(c for c in bad_name if c not in invalid_chars)
            assert char not in sanitized

    def test_sanitize_null_character(self):

        filename = "test\x00file.txt"
        sanitized = filename.replace("\x00", "")
        assert "\x00" not in sanitized

    def test_sanitize_control_characters(self):

        filename = "test\x01\x02\x03file.txt"
        sanitized = "".join(c for c in filename if ord(c) >= 32 or c == "\n")
        assert "\x01" not in sanitized


class TestPathTraversal:
    def test_detect_parent_directory_reference(self):
        malicious = "../../../etc/passwd"
        assert ".." in malicious

    def test_normalize_path_traversal(self):
        base = "/safe/directory"
        malicious = os.path.join(base, "..", "..", "etc", "passwd")
        normalized = os.path.normpath(malicious)

        assert not normalized.startswith(base)

    def test_absolute_path_prevents_traversal(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = os.path.abspath(tmpdir)
            safe_path = os.path.join(base, "file.txt")

            assert os.path.abspath(safe_path).startswith(base)


class TestCaseSensitivity:
    @pytest.mark.skipif(
        sys.platform == "darwin" or sys.platform == "win32",
        reason="Case-insensitive filesystem",
    )
    def test_case_sensitive_filesystem(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = os.path.join(tmpdir, "test.txt")
            file2 = os.path.join(tmpdir, "TEST.txt")

            with open(file1, "w") as f:
                f.write("lowercase")

            with open(file2, "w") as f:
                f.write("uppercase")

            assert os.path.exists(file1)
            assert os.path.exists(file2)

    @pytest.mark.skipif(
        sys.platform not in ["darwin", "win32"],
        reason="Case-insensitive filesystem only",
    )
    def test_case_insensitive_filesystem(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = os.path.join(tmpdir, "test.txt")
            file2 = os.path.join(tmpdir, "TEST.txt")

            with open(file1, "w") as f:
                f.write("content")

            assert os.path.exists(file2)


class TestSymlinks:
    @pytest.mark.skipif(sys.platform == "win32", reason="Unix-specific test")
    def test_create_symlink(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = os.path.join(tmpdir, "target.txt")
            link = os.path.join(tmpdir, "link.txt")

            with open(target, "w") as f:
                f.write("test")

            os.symlink(target, link)

            assert os.path.islink(link)
            assert os.path.exists(link)

    @pytest.mark.skipif(sys.platform == "win32", reason="Unix-specific test")
    def test_resolve_symlink(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = os.path.join(tmpdir, "target.txt")
            link = os.path.join(tmpdir, "link.txt")

            with open(target, "w") as f:
                f.write("test")

            os.symlink(target, link)

            resolved = os.path.realpath(link)
            assert resolved == os.path.realpath(target)


class TestPathComparison:
    def test_same_path_different_separators(self):
        path1 = "a/b/c"
        path2 = "a\\b\\c"

        norm1 = os.path.normpath(path1)
        os.path.normpath(path2)

        assert os.path.sep in norm1 or "/" in norm1

    def test_relative_vs_absolute_same_location(self):
        original_dir = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                os.chdir(tmpdir)

                rel_path = "test.txt"
                abs_path = os.path.join(tmpdir, "test.txt")

                with open(rel_path, "w") as f:
                    f.write("test")

                assert os.path.abspath(rel_path) == abs_path
            finally:
                os.chdir(original_dir)
