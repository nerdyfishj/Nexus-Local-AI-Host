"""Create a .cmdAI launcher on the current user's desktop."""

import os
import shlex
import struct
import subprocess
import sys
import sysconfig
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "cmdai.py"


def icon_path():
    source_icon = ROOT / "cmdAI.ico"
    if source_icon.is_file():
        return source_icon
    return Path(sysconfig.get_path("data")) / "share" / "cmdai" / "cmdAI.ico"


def png_data_from_icon(icon_path):
    icon_data = Path(icon_path).read_bytes()
    if len(icon_data) < 22:
        raise ValueError("icon file is truncated")

    reserved, image_type, image_count = struct.unpack("<HHH", icon_data[:6])
    if reserved != 0 or image_type != 1 or image_count < 1:
        raise ValueError("file is not a valid Windows icon")

    entry = struct.unpack("<BBBBHHII", icon_data[6:22])
    image_size, image_offset = entry[6:8]
    image_end = image_offset + image_size
    if image_offset < 22 or image_end > len(icon_data):
        raise ValueError("icon image data is truncated")

    png_data = icon_data[image_offset:image_end]
    if not png_data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("icon does not contain a PNG image")
    return png_data


def write_logo_png(image_path):
    image_path.write_bytes(png_data_from_icon(icon_path()))


def desktop_directory():
    if sys.platform == "win32":
        result = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", "[Environment]::GetFolderPath('Desktop')"],
            text=True,
        ).strip()
        return Path(result)
    return Path(os.environ.get("XDG_DESKTOP_DIR", str(Path.home() / "Desktop"))).expanduser()


def install_windows(desktop):
    shortcut = desktop / ".cmdAI.lnk"
    icon = icon_path()
    quote = lambda value: "'{}'".format(str(value).replace("'", "''"))
    command = (
        "$shell = New-Object -ComObject WScript.Shell; "
        "$link = $shell.CreateShortcut({}); "
        "$link.TargetPath = {}; $link.Arguments = {}; "
        "$link.WorkingDirectory = {}; $link.IconLocation = {}; $link.Save()"
    ).format(
        quote(shortcut),
        quote(sys.executable),
        quote('"{}"'.format(SCRIPT)),
        quote(ROOT),
        quote("{},0".format(icon)),
    )
    subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
        check=True,
    )
    return shortcut


def install_unix(desktop):
    if sys.platform == "darwin":
        shortcut = desktop / "cmdAI.app"
        contents_dir = shortcut / "Contents"
        executable_dir = contents_dir / "MacOS"
        executable_dir.mkdir(parents=True, exist_ok=True)
        terminal_command = "cd {} && {} {}".format(
            shlex.quote(str(ROOT)), shlex.quote(sys.executable), shlex.quote(str(SCRIPT))
        )
        escape_applescript = lambda value: str(value).replace("\\", "\\\\").replace('"', '\\"')
        applescript = 'tell application "Terminal" to do script "{}"'.format(
            escape_applescript(terminal_command)
        )
        launcher = executable_dir / "launcher"
        launcher.write_text(
            "#!/bin/sh\nosascript -e {} -e {}\n".format(
                shlex.quote(applescript), shlex.quote('tell application "Terminal" to activate')
            ),
            encoding="utf-8",
        )
        launcher.chmod(0o755)
        (contents_dir / "Info.plist").write_text(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
            "<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" "
            "\"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n"
            "<plist version=\"1.0\"><dict>"
            "<key>CFBundleExecutable</key><string>launcher</string>"
            "<key>CFBundleIdentifier</key><string>local.cmdai.launcher</string>"
            "<key>CFBundleName</key><string>.cmdAI</string>"
            "<key>CFBundleDisplayName</key><string>.cmdAI</string>"
            "<key>CFBundlePackageType</key><string>APPL</string>"
            "<key>CFBundleVersion</key><string>1.0</string>"
            "</dict></plist>\n",
            encoding="utf-8",
        )
    else:
        shortcut = desktop / "cmdAI.desktop"
        logo_path = ROOT / "cmdAI.png"
        if not (ROOT / "cmdAI.ico").is_file():
            logo_path = desktop / "cmdAI.png"
        write_logo_png(logo_path)
        escape = lambda value: str(value).replace("\\", "\\\\").replace('"', '\\"')
        contents = (
            "[Desktop Entry]\nType=Application\nName=.cmdAI\nTerminal=true\n"
            "Exec=\"{}\" \"{}\"\nIcon=\"{}\"\nCategories=Utility;\n"
        ).format(escape(sys.executable), escape(SCRIPT), escape(logo_path))
        shortcut.write_text(contents, encoding="utf-8")
        shortcut.chmod(0o755)
    return shortcut


def main():
    desktop = desktop_directory()
    desktop.mkdir(parents=True, exist_ok=True)
    shortcut = install_windows(desktop) if sys.platform == "win32" else install_unix(desktop)
    print("Created desktop launcher: {}".format(shortcut))
    print("The default Ollama provider runs locally and does not require an API key.")


if __name__ == "__main__":
    main()