import re
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "install.bat"
RAW = SCRIPT.read_bytes()
LINES = RAW.decode("utf-8").split("\r\n")
RUN = re.compile(r'^start /wait "" "%(\w+)%"')


def installer_runs():
    return [
        (index, RUN.match(line).group(1))
        for index, line in enumerate(LINES)
        if RUN.match(line)
    ]


def check_before(index):
    return LINES[index - 2], LINES[index - 1]


def signer_for(url_fragment):
    download = next(i for i, line in enumerate(LINES) if url_fragment in line)
    run = next(i for i, _ in installer_runs() if i > download)
    return check_before(run)[0]


def test_every_downloaded_installer_is_checked_right_before_it_runs():
    runs = installer_runs()
    assert len(runs) == 3
    for index, variable in runs:
        check, guard = check_before(index)
        assert check.startswith(f'call :verify_signature "%{variable}%" ')
        assert guard.startswith("if errorlevel 1 (")
        assert "goto end_error" in guard


def test_the_python_installer_must_be_signed_by_the_python_foundation():
    assert signer_for("python-3.11.5-amd64.exe").endswith(
        '"Python Software Foundation"'
    )


def test_the_compiler_installer_must_be_signed_by_the_llvm_foundation():
    assert signer_for("LLVM-18.1.8-win64.exe").endswith('"LLVM Foundation"')


def test_the_windows_sdk_installer_must_be_signed_by_microsoft():
    assert signer_for("go.microsoft.com").endswith('"Microsoft Corporation"')


def verify_routine():
    start = LINES.index(":verify_signature")
    end = LINES.index("", start)
    return LINES[start:end]


def test_the_check_requires_a_valid_signature_from_the_expected_signer():
    body = "\n".join(verify_routine())
    assert "Get-AuthenticodeSignature" in body
    assert "-ne 'Valid'" in body
    assert "SignerCertificate.Subject" in body
    assert "exit /b 1" in body


def test_the_file_path_reaches_powershell_through_the_environment():
    body = "\n".join(verify_routine())
    assert "$env:KORSHMER_VERIFY_PATH" in body
    assert "$env:KORSHMER_VERIFY_SIGNER" in body
    powershell = next(l for l in verify_routine() if "powershell" in l)
    assert "%~1" not in powershell and "%~2" not in powershell


def test_system_settings_change_only_after_the_compiler_download_is_checked():
    setx = next(
        i for i, l in enumerate(LINES) if "setx /M LLVM_INSTALL_DIR" in l
    )
    check = next(
        i
        for i, l in enumerate(LINES)
        if l.startswith("call :verify_signature") and "LLVM Foundation" in l
    )
    assert check < setx


def test_the_script_keeps_windows_line_endings_as_a_safeguard():
    assert b"\r\n" in RAW
    assert b"\n" not in RAW.replace(b"\r\n", b"")
