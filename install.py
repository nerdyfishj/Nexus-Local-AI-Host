"""Create a .cmdAI launcher on the current user's desktop."""

import binascii
import os
import shlex
import struct
import subprocess
import sys
import zlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "cmdai.py"


def write_windows_icon(icon_path):
    size = 128
    pixels = bytearray(size * size * 4)

    def paint(pixel_x, pixel_y, color):
        if 0 <= pixel_x < size and 0 <= pixel_y < size:
            offset = (pixel_y * size + pixel_x) * 4
            pixels[offset:offset + 4] = bytes(color)

    def fill_rounded_rectangle(left, top, right, bottom, radius, color):
        for pixel_y in range(max(0, top), min(size, bottom + 1)):
            for pixel_x in range(max(0, left), min(size, right + 1)):
                nearest_x = min(max(pixel_x, left + radius), right - radius)
                nearest_y = min(max(pixel_y, top + radius), bottom - radius)
                distance_x = pixel_x - nearest_x
                distance_y = pixel_y - nearest_y
                if distance_x * distance_x + distance_y * distance_y <= radius * radius:
                    paint(pixel_x, pixel_y, color)

    def draw_segment(start_x, start_y, end_x, end_y, width, color):
        delta_x = end_x - start_x
        delta_y = end_y - start_y
        length_squared = delta_x * delta_x + delta_y * delta_y
        radius = width / 2
        for pixel_y in range(max(0, int(min(start_y, end_y) - radius)), min(size, int(max(start_y, end_y) + radius + 1))):
            for pixel_x in range(max(0, int(min(start_x, end_x) - radius)), min(size, int(max(start_x, end_x) + radius + 1))):
                fraction = ((pixel_x - start_x) * delta_x + (pixel_y - start_y) * delta_y) / length_squared
                fraction = max(0, min(1, fraction))
                nearest_x = start_x + fraction * delta_x
                nearest_y = start_y + fraction * delta_y
                distance_x = pixel_x - nearest_x
                distance_y = pixel_y - nearest_y
                if distance_x * distance_x + distance_y * distance_y <= radius * radius:
                    paint(pixel_x, pixel_y, color)

    def draw_circle(center_x, center_y, radius, color):
        for pixel_y in range(max(0, int(center_y - radius)), min(size, int(center_y + radius + 1))):
            for pixel_x in range(max(0, int(center_x - radius)), min(size, int(center_x + radius + 1))):
                distance_x = pixel_x - center_x
                distance_y = pixel_y - center_y
                if distance_x * distance_x + distance_y * distance_y <= radius * radius:
                    paint(pixel_x, pixel_y, color)

    fill_rounded_rectangle(4, 4, 123, 123, 27, (35, 218, 205, 255))
    fill_rounded_rectangle(9, 9, 118, 118, 23, (14, 27, 39, 255))

    cyan = (91, 235, 226, 255)
    green = (170, 244, 108, 255)
    draw_segment(79, 34, 94, 34, 4, green)
    draw_segment(94, 34, 94, 46, 4, green)
    draw_segment(78, 48, 86, 48, 4, cyan)
    draw_segment(86, 48, 86, 55, 4, cyan)
    draw_circle(79, 34, 3.5, cyan)
    draw_circle(94, 34, 3.5, green)
    draw_circle(86, 55, 2.5, green)

    draw_segment(34, 52, 54, 65, 8, cyan)
    draw_segment(54, 65, 34, 79, 8, cyan)
    draw_segment(64, 80, 94, 80, 8, green)

    scanlines = b"".join(
        b"\x00" + bytes(pixels[row * size * 4:(row + 1) * size * 4])
        for row in range(size)
    )

    def png_chunk(chunk_type, data):
        checksum = binascii.crc32(chunk_type + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", checksum)

    png_data = (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0))
        + png_chunk(b"IDAT", zlib.compress(scanlines))
        + png_chunk(b"IEND", b"")
    )
    icon_directory = struct.pack("<HHH", 0, 1, 1)
    icon_entry = struct.pack("<BBBBHHII", size, size, 0, 0, 1, 32, len(png_data), 22)
    icon_path.write_bytes(icon_directory + icon_entry + png_data)


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
    icon_path = ROOT / "cmdAI.ico"
    write_windows_icon(icon_path)
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
        quote("{},0".format(icon_path)),
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
        escape = lambda value: str(value).replace("\\", "\\\\").replace('"', '\\"')
        contents = (
            "[Desktop Entry]\nType=Application\nName=.cmdAI\nTerminal=true\n"
            "Exec=\"{}\" \"{}\"\nCategories=Utility;\n"
        ).format(escape(sys.executable), escape(SCRIPT))
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