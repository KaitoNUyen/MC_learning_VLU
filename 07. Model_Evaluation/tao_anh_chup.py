# -*- coding: utf-8 -*-
"""Script tự động chạy 6 bài tập trong baitap07/ và tạo ảnh chụp màn hình terminal PowerShell."""

import os
import subprocess
import sys
from PIL import Image, ImageDraw, ImageFont

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BAITAP_DIR = os.path.join(BASE_DIR, "baitap07")

FONT_PATH = "C:/Windows/Fonts/consola.ttf"
FONT_BOLD_PATH = "C:/Windows/Fonts/consolab.ttf"
FONT_SIZE = 15
LINE_HEIGHT = 21
PADDING_LEFT = 24
PADDING_TOP = 50
PADDING_BOTTOM = 25

BG_COLOR = (12, 12, 12)
HEADER_COLOR = (31, 31, 31)
TEXT_COLOR = (220, 220, 220)
COMMAND_COLOR = (255, 215, 0)
TITLE_COLOR = (200, 200, 200)
PROMPT_COLOR = (90, 200, 255)
ACCENT_GREEN = (90, 220, 110)
ACCENT_YELLOW = (255, 215, 70)
ACCENT_CYAN = (120, 220, 255)

font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
font_bold = ImageFont.truetype(FONT_BOLD_PATH, FONT_SIZE)
font_title = ImageFont.truetype(FONT_PATH, 13)


def render_terminal_screenshot(command_str, output_lines, save_path):
    """Vẽ ảnh chụp cửa sổ Windows PowerShell hiển thị lệnh và toàn bộ kết quả."""
    prompt_prefix = "PS D:\\ML_ungdung\\07. Model_Evaluation> "
    all_lines = [("prompt", prompt_prefix + command_str)]
    for line in output_lines:
        all_lines.append(("text", line))

    # Tính kích thước ảnh vừa khít nội dung
    dummy = Image.new("RGB", (10, 10))
    dummy_draw = ImageDraw.Draw(dummy)
    max_width_px = max(
        dummy_draw.textlength(text, font=font_bold if kind == "prompt" else font)
        for kind, text in all_lines
    )
    img_width = max(920, int(max_width_px) + PADDING_LEFT * 2 + 12)
    img_height = PADDING_TOP + len(all_lines) * LINE_HEIGHT + PADDING_BOTTOM

    img = Image.new("RGB", (img_width, img_height), color=BG_COLOR)
    draw = ImageDraw.Draw(img)

    # Thanh tiêu đề cửa sổ Windows PowerShell
    draw.rectangle([0, 0, img_width, 36], fill=HEADER_COLOR)
    draw.text(
        (16, 10),
        "Windows PowerShell - D:\\ML_ungdung\\07. Model_Evaluation",
        font=font_title,
        fill=TITLE_COLOR,
    )
    draw.line([img_width - 110, 18, img_width - 98, 18], fill=(180, 180, 180), width=1)
    draw.rectangle(
        [img_width - 75, 12, img_width - 63, 24],
        outline=(180, 180, 180),
        width=1,
    )
    draw.line([img_width - 35, 12, img_width - 23, 24], fill=(220, 80, 80), width=2)
    draw.line([img_width - 35, 24, img_width - 23, 12], fill=(220, 80, 80), width=2)

    # Vẽ từng dòng nội dung
    y_pos = PADDING_TOP
    for kind, text in all_lines:
        if kind == "prompt":
            draw.text((PADDING_LEFT, y_pos), prompt_prefix, font=font_bold, fill=PROMPT_COLOR)
            p_len = draw.textlength(prompt_prefix, font=font_bold)
            draw.text((PADDING_LEFT + p_len, y_pos), command_str, font=font_bold, fill=COMMAND_COLOR)
        else:
            color = TEXT_COLOR
            if text.startswith("===="):
                color = ACCENT_GREEN
            elif text.startswith("BAI TAP "):
                color = ACCENT_YELLOW
            elif text.startswith("----") or text.startswith("  ----"):
                color = (140, 140, 140)
            elif any(
                key in text
                for key in (
                    "KHOP HOAN TOAN",
                    "CO DAT (>= 0.80)",
                    "=> Nguong LON NHAT",
                    "Mo hinh thay doi nhieu nhat:",
                    "Da luu hinh ROC tai",
                    "DAT  <-- CHON",
                )
            ):
                color = ACCENT_GREEN
            elif text.startswith(
                (
                    "1. ",
                    "2. ",
                    "3. ",
                    "MO HINH:",
                    "CLASSIFICATION REPORT",
                    "CAC ANH ",
                    "TRA LOI",
                    "GIAI THICH",
                    "PHAN TICH",
                    "NHAN XET",
                )
            ):
                color = ACCENT_CYAN
            draw.text((PADDING_LEFT, y_pos), text, font=font, fill=color)
        y_pos += LINE_HEIGHT

    img.save(save_path)
    print(f"Da tao anh chup: {save_path}")


def chay_va_chup(cmd, ten_anh):
    res = subprocess.run(
        cmd,
        cwd=BASE_DIR,
        shell=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if res.returncode != 0:
        print(f"LOI khi chay {cmd}:\n{res.stderr}")
        sys.exit(res.returncode)
    lines = res.stdout.rstrip().split("\n")
    save_path = os.path.join(BAITAP_DIR, ten_anh)
    render_terminal_screenshot(cmd, lines, save_path)
    return res.stdout


def main():
    for i in range(1, 7):
        chay_va_chup(f"python baitap07/bai{i}.py", f"screenshot_bai{i}.png")


if __name__ == "__main__":
    main()
