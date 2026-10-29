from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent
README_PATH = ROOT_DIR / "README.md"


class TestReadmeExists:
    def test_readme_exists(self):
        assert README_PATH.exists(), "README.md does not exist"

    def test_readme_not_empty(self):
        content = README_PATH.read_text(encoding="utf-8")
        assert len(content) > 0, "README.md is empty"


class TestReadmeFluidSynthRights:
    def _get_readme_content(self):
        return README_PATH.read_text(encoding="utf-8")

    def test_mentions_fluidsynth_directory(self):
        content = self._get_readme_content()
        assert "fluidsynth/" in content or "fluidsynth`" in content, (
            "README.md should mention the fluidsynth/ directory"
        )

    def test_mentions_fluidsynth_exe(self):
        content = self._get_readme_content()
        assert "fluidsynth.exe" in content, (
            "README.md should mention fluidsynth.exe"
        )

    def test_mentions_libfluidsynth_dll(self):
        content = self._get_readme_content()
        assert "libfluidsynth-3.dll" in content, (
            "README.md should mention libfluidsynth-3.dll"
        )

    def test_mentions_sdl3_dll(self):
        content = self._get_readme_content()
        assert "SDL3.dll" in content, "README.md should mention SDL3.dll"

    def test_mentions_sndfile_dll(self):
        content = self._get_readme_content()
        assert "sndfile.dll" in content, "README.md should mention sndfile.dll"

    def test_mentions_soundfont_sf3(self):
        content = self._get_readme_content()
        assert "soundfont.sf3" in content, (
            "README.md should mention soundfont.sf3"
        )

    def test_mentions_lgpl_license(self):
        content = self._get_readme_content()
        assert "LGPL" in content, (
            "README.md should mention LGPL license for FluidSynth"
        )

    def test_mentions_zlib_license(self):
        content = self._get_readme_content()
        assert "zlib" in content, (
            "README.md should mention zlib license for SDL3"
        )

    def test_mentions_third_party_licenses(self):
        content = self._get_readme_content()
        assert "Third-Party" in content, (
            "README.md should have a Third-Party Licenses section"
        )

    def test_all_fluidsynth_files_documented(self):
        content = self._get_readme_content()
        fluidsynth_dir = ROOT_DIR / "fluidsynth"
        expected_files = [
            "fluidsynth.exe",
            "libfluidsynth-3.dll",
            "SDL3.dll",
            "sndfile.dll",
            "soundfont.sf3",
        ]
        if fluidsynth_dir.exists():
            for name in expected_files:
                assert name in content, (
                    f"fluidsynth/{name} is not documented in README.md"
                )


class TestReadmeStructure:
    def _get_readme_content(self):
        return README_PATH.read_text(encoding="utf-8")

    def test_has_title(self):
        content = self._get_readme_content()
        assert content.startswith("#"), "README.md should start with a heading"

    def test_has_license_section(self):
        content = self._get_readme_content()
        assert "License" in content, "README.md should have a License section"

    def test_has_installation_section(self):
        content = self._get_readme_content()
        assert "Installation" in content, (
            "README.md should have an Installation section"
        )

    def test_has_testing_section(self):
        content = self._get_readme_content()
        assert "Testing" in content, "README.md should have a Testing section"
