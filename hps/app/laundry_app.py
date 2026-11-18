#!/usr/bin/env python3
"""Standalone DE10-Standard HPS app: removable-media browser, preprocessing,
FPGA CNN driver, persistent inventory, and 640x480 RGB565 VGA framebuffer.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import mmap
import os
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from algoritma.preprocessing import preprocess_bgr  # noqa: E402

CLASS_NAMES = ["Baju", "Celana", "Jaket", "Celana Dalam", "Kaus Dalam"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SCREEN_W, SCREEN_H = 640, 480
FB_BYTES = SCREEN_W * SCREEN_H * 2
CNN_REG = {"CTRL": 0x00, "STATUS": 0x04, "GEOMETRY": 0x08,
           "RESULT": 0x0C, "CYCLES": 0x10, "MASK": 0x100}
BUTTON_REG = {"STATUS": 0x00, "EVENTS": 0x04, "SWITCHES": 0x08}


@dataclass(frozen=True)
class MediaImage:
    path: Path
    source: str


def load_config(path: Path, require_vga: bool = True, require_buttons: bool = True) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Board config tidak ditemukan: {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    required = ["cnn_cpu_phys_base"] + (["buttons_cpu_phys_base"] if require_buttons else [])
    if require_vga:
        required.append("framebuffer_cpu_phys_base")
    for key in required:
        if key not in config or config[key] in (None, "", "0x00000000", 0):
            raise ValueError(f"Alamat {key} belum diisi dari system.sopcinfo/device tree hasil Platform Designer.")
    return config


def mounted_media() -> list[tuple[Path, str]]:
    found: dict[str, tuple[Path, str]] = {}
    try:
        lines = Path("/proc/mounts").read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        fields = line.split()
        if len(fields) < 2:
            continue
        device, mount_text = fields[0], fields[1].replace("\\040", " ")
        mount = Path(mount_text)
        if not mount.is_dir():
            continue
        low = mount_text.lower()
        # Never recurse through system mounts (rootfs, boot, proc, sys, dev).
        if mount == Path("/") or low.startswith(("/proc", "/sys", "/dev", "/boot")):
            continue
        is_removable_mount = low.startswith(("/media/", "/run/media/", "/mnt/"))
        if device.startswith("/dev/mmcblk") and is_removable_mount:
            kind = "sd"
        elif device.startswith("/dev/sd") and is_removable_mount:
            kind = "usb"
        elif is_removable_mount and ("microsd" in low or "sdcard" in low):
            kind = "sd"
        elif is_removable_mount:
            kind = "unknown"
        else:
            continue
        found[str(mount)] = (mount, kind)
    return list(found.values())


def find_images(source_mode: str = "both", explicit_roots: Iterable[Path] = ()) -> list[MediaImage]:
    roots = [(Path(p).resolve(), "manual") for p in explicit_roots]
    if not roots:
        roots.extend(mounted_media())
    selected: list[MediaImage] = []
    seen: set[str] = set()
    for mount, kind in roots:
        if source_mode != "both" and kind not in (source_mode, "manual"):
            continue
        preferred = [mount / "LAUNDRY" / "images", mount / "images", mount]
        image_root = next((p for p in preferred if p.is_dir() and any(
            x.is_file() and x.suffix.lower() in IMAGE_EXTENSIONS for x in p.iterdir())), mount)
        for path in sorted(image_root.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                key = str(path.resolve())
                if key not in seen:
                    seen.add(key)
                    selected.append(MediaImage(path.resolve(), kind))
    return selected


class MmioRegion:
    """Initial /dev/mem mapping; replace with UIO/driver for production."""
    def __init__(self, base: int, size: int = 4096, device: str = "/dev/mem"):
        page = mmap.PAGESIZE
        self.aligned_base = base & ~(page - 1)
        self.delta = base - self.aligned_base
        self.size = ((self.delta + size + page - 1) // page) * page
        self.fd = os.open(device, os.O_RDWR | os.O_SYNC)
        try:
            self.map = mmap.mmap(self.fd, self.size, flags=mmap.MAP_SHARED,
                                 prot=mmap.PROT_READ | mmap.PROT_WRITE,
                                 offset=self.aligned_base)
        except Exception:
            os.close(self.fd)
            raise

    def read32(self, offset: int) -> int:
        return struct.unpack_from("<I", self.map, self.delta + offset)[0]

    def write32(self, offset: int, value: int) -> None:
        struct.pack_into("<I", self.map, self.delta + offset, value & 0xFFFFFFFF)

    def write_bytes(self, offset: int, payload: bytes) -> None:
        self.map[self.delta + offset:self.delta + offset + len(payload)] = payload

    def close(self) -> None:
        self.map.close()
        os.close(self.fd)


class FpgaCNN:
    """Driver for cnn_avalon_slave.v; these are HPS CPU byte offsets."""
    def __init__(self, cpu_phys_base: int, timeout_s: float = 10.0):
        self.mmio = MmioRegion(cpu_phys_base, size=0x400)
        self.timeout_s = timeout_s

    @staticmethod
    def pack_mask(mask: np.ndarray) -> list[int]:
        binary = (mask.reshape(-1) > 0).astype(np.uint8)
        packed = np.packbits(binary, bitorder="little")
        if packed.size != 512:
            raise ValueError(f"Mask wajib 64x64 (512 byte), didapat {packed.size} byte.")
        return [int.from_bytes(packed[i:i+4].tobytes(), "little") for i in range(0, 512, 4)]

    def infer(self, mask: np.ndarray, geometry: np.ndarray) -> tuple[int, int]:
        if mask.shape != (64, 64) or geometry.shape != (4,):
            raise ValueError("Input FPGA harus mask 64x64 dan 4 fitur geometri.")
        for index, word in enumerate(self.pack_mask(mask)):
            self.mmio.write32(CNN_REG["MASK"] + 4 * index, word)
        geom_q7 = [int(np.clip(np.rint(float(v) * 127.0), 0, 127)) for v in geometry]
        geometry_word = sum(v << (8 * i) for i, v in enumerate(geom_q7))
        self.mmio.write32(CNN_REG["GEOMETRY"], geometry_word)
        self.mmio.write32(CNN_REG["CTRL"], 1)
        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            status = self.mmio.read32(CNN_REG["STATUS"])
            if status & 0x2:
                result = self.mmio.read32(CNN_REG["RESULT"]) & 0x7
                cycles = self.mmio.read32(CNN_REG["CYCLES"])
                if result > 4:
                    raise RuntimeError(f"Class ID tidak valid: {result}")
                return result, cycles
            time.sleep(0.001)
        raise TimeoutError("CNN timeout; periksa bitstream, reset, clock, alamat CSR, dan bobot ROM.")

    def close(self) -> None:
        self.mmio.close()


class UserButtons:
    def __init__(self, cpu_phys_base: int):
        self.mmio = MmioRegion(cpu_phys_base, size=4096)

    def poll(self) -> tuple[int, int]:
        events = self.mmio.read32(BUTTON_REG["EVENTS"]) & 0xF
        switches = self.mmio.read32(BUTTON_REG["SWITCHES"]) & 0x3FF
        if events:
            self.mmio.write32(BUTTON_REG["EVENTS"], events)
        return events, switches

    def close(self) -> None:
        self.mmio.close()


class Inventory:
    def __init__(self, path: Path):
        self.path = path
        self.data = {"version": 1, "items": []}
        if path.exists():
            self.data = json.loads(path.read_text(encoding="utf-8"))
            self.data.setdefault("items", [])

    def add(self, image: Path, class_id: int, source: str) -> bool:
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        if any(x.get("sha256") == digest for x in self.data["items"]):
            return False
        self.data["items"].append({
            "sha256": digest, "filename": image.name, "path": str(image),
            "source": source, "class_id": class_id, "class_name": CLASS_NAMES[class_id],
            "added_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")
        })
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        temp.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(temp, self.path)
        return True

    def counts(self) -> list[int]:
        result = [0] * len(CLASS_NAMES)
        for item in self.data["items"]:
            cid = int(item.get("class_id", -1))
            if 0 <= cid < len(result):
                result[cid] += 1
        return result


def choose_inventory_path(config: dict) -> Path:
    if config.get("inventory_file"):
        return Path(config["inventory_file"])
    for mount, kind in mounted_media():
        if kind == "sd" and os.access(mount, os.W_OK):
            return mount / "LAUNDRY" / "inventory.json"
    return Path.home() / "LAUNDRY" / "inventory.json"


class VgaFramebuffer:
    """Write 640x480 RGB565 to shared FPGA SDRAM through the HPS-to-FPGA window."""
    def __init__(self, cpu_phys_base: int):
        self.mmio = MmioRegion(cpu_phys_base, size=FB_BYTES)

    def show(self, image: Image.Image) -> None:
        rgb = np.asarray(image.convert("RGB").resize((SCREEN_W, SCREEN_H)), dtype=np.uint16)
        r, g, b = rgb[:, :, 0] >> 3, rgb[:, :, 1] >> 2, rgb[:, :, 2] >> 3
        pixels = ((r << 11) | (g << 5) | b).astype("<u2")
        self.mmio.write_bytes(0, pixels.tobytes(order="C"))

    def close(self) -> None:
        self.mmio.close()


def render_screen(current, result, inventory, index, total, message) -> Image.Image:
    canvas = Image.new("RGB", (SCREEN_W, SCREEN_H), (12, 20, 34))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.rectangle((0, 0, SCREEN_W, 54), fill=(27, 44, 66))
    draw.text((18, 17), "SMART LAUNDRY  |  DE10-STANDARD", fill="white", font=font)
    draw.text((500, 17), "STANDALONE", fill=(91, 220, 145), font=font)
    draw.rectangle((16, 70, 342, 410), outline=(77, 101, 132), width=2)
    if current and current.path.exists():
        try:
            with Image.open(current.path) as src:
                src.thumbnail((310, 320))
                canvas.paste(src.convert("RGB"), (179-src.width//2, 240-src.height//2))
        except Exception as exc:
            draw.text((28, 100), f"Gagal membuka gambar: {exc}", fill=(255, 120, 120), font=font)
    else:
        draw.text((92, 220), "Tidak ada foto", fill=(210, 220, 232), font=font)
    draw.text((20, 422), f"Foto {index+1 if total else 0}/{total}", fill=(210, 220, 232), font=font)
    if current:
        draw.text((365, 78), current.path.name[:34], fill=(240, 245, 250), font=font)
        draw.text((365, 108), f"Sumber: {current.source.upper()}", fill=(165, 184, 207), font=font)
    draw.text((365, 150), "HASIL KLASIFIKASI", fill=(165, 184, 207), font=font)
    if result:
        draw.text((365, 176), CLASS_NAMES[result["class_id"]].upper(), fill=(91, 220, 145), font=font)
        draw.text((365, 202), f"Class ID: {result['class_id']}", fill=(235, 240, 248), font=font)
        draw.text((365, 226), f"RTL cycles: {result['cycles']}", fill=(235, 240, 248), font=font)
        draw.text((365, 250), "Confidence not exposed by ArgMax RTL", fill=(255, 205, 120), font=font)
    else:
        draw.text((365, 180), "Belum diklasifikasi", fill=(230, 230, 230), font=font)
    draw.rectangle((360, 288, 622, 411), outline=(77, 101, 132), width=2)
    draw.text((374, 300), "INVENTARIS", fill=(91, 220, 145), font=font)
    counts = inventory.counts()
    for i, name in enumerate(CLASS_NAMES):
        draw.text((374, 322+i*16), f"{name:<15} {counts[i]:>4}", fill=(238, 243, 249), font=font)
    draw.rectangle((0, 450, SCREEN_W, 479), fill=(27, 44, 66))
    draw.text((16, 459), "KEY0 PREV | KEY1 NEXT | KEY2 CLASSIFY | KEY3 ADD", fill=(245, 245, 245), font=font)
    draw.text((16, 436), message[:85], fill=(255, 205, 120), font=font)
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone DE10-Standard Smart Laundry HPS application")
    parser.add_argument("--config", type=Path, default=ROOT / "hps/config/de10_standard.json")
    parser.add_argument("--source", choices=("both", "usb", "sd"), default="both")
    parser.add_argument("--no-vga", action="store_true")
    parser.add_argument("--keyboard", action="store_true", help="Host test only: p/n/c/a/q")
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()
    cfg = load_config(args.config, require_vga=not args.no_vga, require_buttons=not args.keyboard)
    roots = [Path(p) for p in cfg.get("media_roots", [])]
    media = find_images(args.source, roots)
    if not media:
        print("Tidak ada JPG/JPEG/PNG pada USB atau MicroSD yang ter-mount.")
        return 2
    inventory = Inventory(choose_inventory_path(cfg))
    cnn = FpgaCNN(int(str(cfg["cnn_cpu_phys_base"]), 0), args.timeout)
    buttons = None if args.keyboard else UserButtons(int(str(cfg["buttons_cpu_phys_base"]), 0))
    framebuffer = None if args.no_vga else VgaFramebuffer(int(str(cfg["framebuffer_cpu_phys_base"]), 0))
    index, current_result = 0, None
    message = f"Menemukan {len(media)} gambar dari USB/MicroSD."
    source_filter = args.source

    def refresh():
        screen = render_screen(media[index], current_result, inventory, index, len(media), message)
        if framebuffer:
            framebuffer.show(screen)
        print(f"[{index+1}/{len(media)}] {media[index].path} | {message}")

    try:
        refresh()
        while True:
            if args.keyboard:
                command = input("n=next p=prev c=classify a=add q=quit > ").strip().lower()
                if command == "q":
                    break
                events, switches = {"p":1,"n":2,"c":4,"a":8}.get(command,0), 0
            else:
                events, switches = buttons.poll()
                time.sleep(0.03)
            requested = "usb" if (switches & 3)==1 else "sd" if (switches & 3)==2 else "both"
            if requested != source_filter:
                source_filter = requested
                filtered = find_images(source_filter, roots)
                if filtered:
                    media = filtered
                    index = min(index, len(media)-1)
                    current_result = None
                    message = f"Filter sumber: {source_filter.upper()} ({len(media)} foto)."
            if events & 1:
                index=(index-1)%len(media); current_result=None; message="Foto sebelumnya."
            if events & 2:
                index=(index+1)%len(media); current_result=None; message="Foto berikutnya."
            if events & 4:
                bgr=cv2.imread(str(media[index].path),cv2.IMREAD_COLOR)
                processed=preprocess_bgr(bgr,size=64)
                if processed is None:
                    current_result=None
                    message="Segmentasi gagal; gunakan foto satu pakaian dengan latar kontras."
                else:
                    cid,cycles=cnn.infer(processed.mask,processed.geometry)
                    current_result={"class_id":cid,"cycles":cycles}
                    message=f"Terdeteksi: {CLASS_NAMES[cid]}. KEY3 untuk simpan."
            if events & 8:
                if current_result is None:
                    message="Belum ada hasil klasifikasi untuk foto ini."
                else:
                    added=inventory.add(media[index].path,current_result["class_id"],media[index].source)
                    message="Barang ditambahkan." if added else "Foto ini sudah terdaftar; tidak dihitung dua kali."
            if events or args.keyboard:
                refresh()
    except KeyboardInterrupt:
        print("\nAplikasi dihentikan.")
    finally:
        cnn.close()
        if buttons:
            buttons.close()
        if framebuffer:
            framebuffer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
