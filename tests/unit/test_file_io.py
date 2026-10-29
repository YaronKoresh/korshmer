import os
import tempfile
from pathlib import Path

import pytest


class TestFilePathValidation:
    def test_absolute_path_validation(self):
        path = "/home/user/test.txt"
        assert os.path.isabs(path)

    def test_relative_path_detection(self):
        path = "relative/path/test.txt"
        assert not os.path.isabs(path)

    def test_path_normalization(self):
        path = "/home/user/../user/./test.txt"
        normalized = os.path.normpath(path)
        assert normalized == os.path.normpath("/home/user/test.txt")

    def test_windows_path_handling(self):
        if os.name == "nt":
            path = "C:\\Users\\test\\file.txt"
            assert os.path.isabs(path)

    def test_unicode_path_handling(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            unicode_name = "тест_测试_テスト.txt"
            file_path = os.path.join(tmpdir, unicode_name)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write("test content")

            assert os.path.exists(file_path)

            with open(file_path, encoding="utf-8") as f:
                content = f.read()
                assert content == "test content"

    def test_special_characters_in_filename(self):

        invalid_chars = '<>:"/\\|?*'

        for char in invalid_chars:
            test_name = f"test{char}file.txt"

            sanitized = "".join(c for c in test_name if c not in invalid_chars)
            assert char not in sanitized

    def test_long_path_handling(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            long_name = "a" * 50 + ".txt"
            file_path = os.path.join(tmpdir, long_name)

            with open(file_path, "w") as f:
                f.write("test")

            assert os.path.exists(file_path)

    def test_reserved_names_windows(self):
        if os.name == "nt":
            reserved = ["CON", "PRN", "AUX", "NUL", "COM1", "LPT1"]
            for name in reserved:
                assert name.upper() in [r.upper() for r in reserved]

    def test_path_traversal_detection(self):
        malicious_paths = ["../../../etc/passwd", "..\\..\\windows\\system32"]

        for path in malicious_paths:
            assert ".." in path

    def test_path_with_spaces(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = "file with spaces.txt"
            file_path = os.path.join(tmpdir, filename)

            with open(file_path, "w") as f:
                f.write("test")

            assert os.path.exists(file_path)


class TestFileExistence:
    def test_file_exists(self):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name

        try:
            assert os.path.exists(tmp_path)
        finally:
            os.unlink(tmp_path)

    def test_file_not_exists(self):
        path = "/nonexistent/path/file.txt"
        assert not os.path.exists(path)

    def test_directory_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            assert os.path.exists(tmpdir)
            assert os.path.isdir(tmpdir)

    def test_is_file_check(self):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name

        try:
            assert os.path.isfile(tmp_path)
            assert not os.path.isdir(tmp_path)
        finally:
            os.unlink(tmp_path)


class TestFilePermissions:
    def test_readable_file(self):
        with tempfile.NamedTemporaryFile(delete=False, mode="w") as tmp:
            tmp.write("test")
            tmp_path = tmp.name

        try:
            assert os.access(tmp_path, os.R_OK)
        finally:
            os.unlink(tmp_path)

    def test_writable_file(self):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name

        try:
            assert os.access(tmp_path, os.W_OK)
        finally:
            os.unlink(tmp_path)

    def test_directory_writable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            assert os.access(tmpdir, os.W_OK)

    @pytest.mark.skipif(os.name == "nt", reason="Unix-specific test")
    @pytest.mark.skipif(
        hasattr(os, "geteuid") and os.geteuid() == 0,
        reason="The superuser can write to read-only files",
    )
    def test_read_only_file(self):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name

        try:
            os.chmod(tmp_path, 0o444)
            assert os.access(tmp_path, os.R_OK)
            assert not os.access(tmp_path, os.W_OK)
        finally:
            os.chmod(tmp_path, 0o644)
            os.unlink(tmp_path)


class TestFileOperations:
    def test_create_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test.txt")

            with open(file_path, "w") as f:
                f.write("test content")

            assert os.path.exists(file_path)

    def test_read_file(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp:
            tmp.write("test content")
            tmp_path = tmp.name

        try:
            with open(tmp_path) as f:
                content = f.read()
            assert content == "test content"
        finally:
            os.unlink(tmp_path)

    def test_write_binary_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test.bin")

            data = b"\x00\x01\x02\x03\xff"
            with open(file_path, "wb") as f:
                f.write(data)

            with open(file_path, "rb") as f:
                read_data = f.read()

            assert read_data == data

    def test_append_to_file(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp:
            tmp.write("line1\n")
            tmp_path = tmp.name

        try:
            with open(tmp_path, "a") as f:
                f.write("line2\n")

            with open(tmp_path) as f:
                content = f.read()

            assert "line1" in content
            assert "line2" in content
        finally:
            os.unlink(tmp_path)

    def test_delete_file(self):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name

        assert os.path.exists(tmp_path)
        os.unlink(tmp_path)
        assert not os.path.exists(tmp_path)


class TestDirectoryOperations:
    def test_create_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = os.path.join(tmpdir, "subdir")
            os.makedirs(new_dir)
            assert os.path.isdir(new_dir)

    def test_create_nested_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_dir = os.path.join(tmpdir, "a", "b", "c")
            os.makedirs(nested_dir, exist_ok=True)
            assert os.path.isdir(nested_dir)

    def test_create_directory_exists_ok(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(tmpdir, exist_ok=True)
            assert os.path.isdir(tmpdir)

    def test_list_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(3):
                file_path = os.path.join(tmpdir, f"file{i}.txt")
                with open(file_path, "w") as f:
                    f.write("test")

            contents = os.listdir(tmpdir)
            assert len(contents) == 3


class TestCorruptedFiles:
    def test_empty_file(self):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with open(tmp_path) as f:
                content = f.read()
            assert content == ""
        finally:
            os.unlink(tmp_path)

    def test_partial_read(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp:
            tmp.write("0123456789")
            tmp_path = tmp.name

        try:
            with open(tmp_path) as f:
                chunk1 = f.read(5)
                chunk2 = f.read(5)

            assert chunk1 == "01234"
            assert chunk2 == "56789"
        finally:
            os.unlink(tmp_path)

    def test_file_encoding_errors(self):
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as tmp:
            tmp.write(b"\xff\xfe")
            tmp_path = tmp.name

        try:
            with open(tmp_path, encoding="utf-8", errors="replace") as f:
                content = f.read()

            assert content is not None
        finally:
            os.unlink(tmp_path)


class TestContextManagers:
    def test_with_statement_closes_file(self):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with open(tmp_path, "w") as f:
                f.write("test")

            with pytest.raises(ValueError):
                f.write("more")
        finally:
            os.unlink(tmp_path)

    def test_exception_in_context_closes_file(self):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name

        try:
            try:
                with open(tmp_path, "w") as f:
                    f.write("test")
                    raise ValueError("Test exception")
            except ValueError:
                pass

            with pytest.raises(ValueError):
                f.write("more")
        finally:
            os.unlink(tmp_path)


class TestPathLib:
    def test_path_object_creation(self):
        path = Path("/home/user/test.txt")
        assert isinstance(path, Path)

    def test_path_join(self):
        base = Path("/home/user")
        file_path = base / "test.txt"
        expected = Path("/home/user/test.txt")
        assert file_path == expected

    def test_path_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            assert path.exists()
            assert path.is_dir()

    def test_path_read_text(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp:
            tmp.write("test content")
            tmp_path = tmp.name

        try:
            path = Path(tmp_path)
            content = path.read_text()
            assert content == "test content"
        finally:
            os.unlink(tmp_path)

    def test_path_write_text(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.txt"
            file_path.write_text("test content")

            assert file_path.exists()
            assert file_path.read_text() == "test content"

    def test_path_suffix(self):
        path = Path("/home/user/test.txt")
        assert path.suffix == ".txt"

    def test_path_stem(self):
        path = Path("/home/user/test.txt")
        assert path.stem == "test"

    def test_path_parent(self):
        path = Path("/home/user/test.txt")
        assert str(path.parent) == os.path.normpath("/home/user")
