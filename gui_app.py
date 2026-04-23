"""
gui_app.py
Premium Blood Cell Classifier GUI — Redesigned with biopunk aesthetic
Deep navy + electric teal + crimson + amber — feels like a medical sci-fi lab dashboard
"""

import sys
import os
import threading
import math
import time
from pathlib import Path
from PIL import Image, ImageTk, ImageFilter, ImageDraw
import tkinter as tk
from tkinter import filedialog, messagebox

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import numpy as np
import customtkinter as ctk

from config import CONFIG
from model import BloodCellCNN
from dataset import get_transforms

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ─── Palette ────────────────────────────────────────────────────────────────
C = {
    "bg":           "#060B14",       # near-black navy
    "panel":        "#0D1B2A",       # deep panel
    "card":         "#111D2E",       # card surface
    "border":       "#1A2F45",       # subtle border
    "border_glow":  "#0E4D6B",       # highlighted border

    "teal":         "#00C2CC",       # electric teal — primary accent
    "teal_dim":     "#00717A",
    "teal_dark":    "#003D42",

    "crimson":      "#E8334A",       # blood red
    "crimson_dim":  "#8B1E2C",

    "amber":        "#FFA94D",       # warm amber — warnings / interesting classes
    "amber_dim":    "#7A4F24",

    "violet":       "#A78BFA",       # soft violet — secondary highlights
    "violet_dim":   "#3D2F6B",

    "green":        "#34D399",       # success / high confidence
    "green_dim":    "#0F5438",

    "text":         "#E2EDF5",
    "text_mid":     "#7FAABF",
    "text_dim":     "#364F63",

    "white":        "#FFFFFF",
}

# Class-specific color palette (vibrant)
CLASS_COLORS = {
    "eosinophil":  C["crimson"],
    "lymphocyte":  C["teal"],
    "monocyte":    C["amber"],
    "neutrophil":  C["violet"],
}

CLASS_ICONS = {
    "eosinophil":  "◈",
    "lymphocyte":  "◉",
    "monocyte":    "◐",
    "neutrophil":  "◍",
}

CLASS_DESC = {
    "eosinophil":  "Allergic & parasitic response",
    "lymphocyte":  "Adaptive immune defense",
    "monocyte":    "Phagocytic & inflammatory",
    "neutrophil":  "First-line bacterial defense",
}


# ─── Canvas Pulse Dot ────────────────────────────────────────────────────────
class PulseDot(tk.Canvas):
    """Animated pulsing status indicator drawn on canvas."""
    def __init__(self, parent, color, size=14, **kwargs):
        super().__init__(parent, width=size, height=size,
                         bg=C["panel"], highlightthickness=0, **kwargs)
        self.color = color
        self.size = size
        self._phase = 0
        self._animate()

    def _animate(self):
        self.delete("all")
        s = self.size
        r = s / 2
        # Outer glow ring (pulsing)
        pulse = 0.45 + 0.35 * math.sin(self._phase)
        pad = r * (1 - pulse)
        self.create_oval(pad, pad, s - pad, s - pad,
                         outline=self.color, width=1.5,
                         stipple="gray50")
        # Inner solid dot
        inner = r * 0.45
        self.create_oval(r - inner, r - inner, r + inner, r + inner,
                         fill=self.color, outline="")
        self._phase += 0.12
        self.after(40, self._animate)

    def set_color(self, color):
        self.color = color


# ─── Animated Gradient Bar ──────────────────────────────────────────────────
class GradientBar(tk.Canvas):
    """
    Smooth animated horizontal progress bar using Canvas.
    Fills left-to-right with a gradient from accent_start → accent_end.
    """
    def __init__(self, parent, height=10, bg_color=C["border"],
                 accent_start=C["teal"], accent_end=C["violet"], **kwargs):
        super().__init__(parent, height=height, bg=C["card"],
                         highlightthickness=0, **kwargs)
        self.bar_bg = bg_color
        self.accent_start = accent_start
        self.accent_end = accent_end
        self._value = 0.0
        self._target = 0.0
        self._animating = False
        self.bind("<Configure>", self._redraw)

    def _hex_to_rgb(self, h):
        h = h.lstrip('#')
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

    def _lerp_color(self, c1, c2, t):
        r1, g1, b1 = self._hex_to_rgb(c1)
        r2, g2, b2 = self._hex_to_rgb(c2)
        r = int(r1 + (r2 - r1) * t)
        g = int(g1 + (g2 - g1) * t)
        b = int(b1 + (b2 - b1) * t)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _redraw(self, event=None):
        self.delete("all")
        w = self.winfo_width()
        h = self.winfo_height()
        if w <= 1:
            return
        r = h // 2
        # Background track
        self.create_rectangle(0, 0, w, h, fill=self.bar_bg, outline="")
        # Fill
        fill_w = int(w * self._value)
        if fill_w > 2:
            steps = max(1, fill_w // 3)
            for i in range(steps):
                t = i / max(steps - 1, 1)
                color = self._lerp_color(self.accent_start, self.accent_end, t)
                x0 = int(fill_w * i / steps)
                x1 = int(fill_w * (i + 1) / steps) + 1
                self.create_rectangle(x0, 1, x1, h - 1, fill=color, outline="")
            # Shine highlight
            self.create_rectangle(1, 1, fill_w - 1, h // 3,
                                   fill="white", outline="", stipple="gray12")

    def set(self, value, animate=True):
        self._target = max(0.0, min(1.0, value))
        if animate and not self._animating:
            self._animating = True
            self._tick()
        else:
            self._value = self._target
            self._redraw()

    def _tick(self):
        diff = self._target - self._value
        if abs(diff) < 0.005:
            self._value = self._target
            self._animating = False
            self._redraw()
            return
        self._value += diff * 0.18
        self._redraw()
        self.after(16, self._tick)


# ─── Hex Badge ───────────────────────────────────────────────────────────────
class HexBadge(tk.Canvas):
    """Draws a hexagon with a colored fill and label — for class icons."""
    def __init__(self, parent, color, icon, size=44, **kwargs):
        super().__init__(parent, width=size, height=size,
                         bg=C["card"], highlightthickness=0, **kwargs)
        self.color = color
        self.icon = icon
        self.size = size
        self._draw()

    def _blend(self, hex_color, bg_hex, alpha=0.15):
        """Blend hex_color onto bg_hex with given alpha (0-1)."""
        def parse(h):
            h = h.lstrip('#')
            return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
        r1, g1, b1 = parse(hex_color)
        r2, g2, b2 = parse(bg_hex)
        r = int(r2 + (r1 - r2) * alpha)
        g = int(g2 + (g1 - g2) * alpha)
        b = int(b2 + (b1 - b2) * alpha)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _draw(self):
        self.delete("all")
        s = self.size
        cx, cy = s / 2, s / 2
        r = s * 0.45
        pts = []
        for i in range(6):
            angle = math.radians(60 * i - 30)
            pts.append(cx + r * math.cos(angle))
            pts.append(cy + r * math.sin(angle))
        fill_color = self._blend(self.color, C["card"], alpha=0.18)
        self.create_polygon(pts, fill=fill_color, outline=self.color,
                            width=1.5, smooth=False)
        self.create_text(cx, cy, text=self.icon, fill=self.color,
                         font=("Courier", 16, "bold"))


# ─── Main App ─────────────────────────────────────────────────────────────────
class BloodCellClassifierGUI:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("HemaScan AI — Blood Cell Classifier")
        self.root.geometry("1360x860")
        self.root.minsize(1100, 720)
        self.root.configure(fg_color=C["bg"])

        self.model = None
        self.transform = None
        self.class_names = CONFIG["class_names"]
        self.device = CONFIG["device"]
        self.current_image_path = None
        self.current_image = None

        self._scan_phase = 0
        self._scanning = False

        self.load_model()
        self.setup_ui()
        self.setup_shortcuts()

    # ── Model ─────────────────────────────────────────────────────────────────
    def load_model(self):
        try:
            model_path = "checkpoints/best.pth"
            if not os.path.exists(model_path):
                messagebox.showwarning("Model Not Found",
                    f"Model not found at {model_path}\n"
                    "Please train the model first:\n  python src/train_main.py")
                return False
            self.model = BloodCellCNN(
                num_classes=CONFIG["num_classes"],
                base_filters=CONFIG["base_filters"],
                dropout_rate=0.0,
            ).to(self.device)
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device))
            self.model.eval()
            self.transform = get_transforms(CONFIG["img_size"], mode="val")
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")
            return False

    # ── UI Shell ──────────────────────────────────────────────────────────────
    def setup_ui(self):
        # Top bar
        self._build_topbar()

        # Body — two columns
        body = tk.Frame(self.root, bg=C["bg"])
        body.pack(fill="both", expand=True, padx=24, pady=(0, 20))
        body.columnconfigure(0, weight=5)
        body.columnconfigure(1, weight=6)
        body.rowconfigure(0, weight=1)

        # Left panel
        left = tk.Frame(body, bg=C["panel"],
                        highlightbackground=C["border"], highlightthickness=1)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        self._build_left(left)

        # Right panel
        right = tk.Frame(body, bg=C["panel"],
                         highlightbackground=C["border"], highlightthickness=1)
        right.grid(row=0, column=1, sticky="nsew")
        self._build_right(right)

        # Footer
        self._build_footer()

    # ── Top Bar ───────────────────────────────────────────────────────────────
    def _build_topbar(self):
        bar = tk.Frame(self.root, bg=C["panel"],
                       highlightbackground=C["border_glow"],
                       highlightthickness=1, height=70)
        bar.pack(fill="x", padx=0, pady=(0, 0))
        bar.pack_propagate(False)

        inner = tk.Frame(bar, bg=C["panel"])
        inner.pack(fill="both", expand=True, padx=28, pady=0)

        # Logo cluster
        logo_frame = tk.Frame(inner, bg=C["panel"])
        logo_frame.pack(side="left", fill="y")

        # Red cell icon (canvas)
        cell_canvas = tk.Canvas(logo_frame, width=40, height=40,
                                bg=C["panel"], highlightthickness=0)
        cell_canvas.pack(side="left", padx=(0, 12), pady=15)
        self._draw_cell_logo(cell_canvas)

        text_block = tk.Frame(logo_frame, bg=C["panel"])
        text_block.pack(side="left", pady=16)

        tk.Label(text_block, text="HemaScan AI", bg=C["panel"],
                 fg=C["teal"], font=("Courier New", 18, "bold")).pack(anchor="w")
        tk.Label(text_block, text="Blood Cell Classification System  v2.0",
                 bg=C["panel"], fg=C["text_mid"],
                 font=("Courier New", 9)).pack(anchor="w")

        # Right cluster — status + shortcuts hint
        right_cluster = tk.Frame(inner, bg=C["panel"])
        right_cluster.pack(side="right", fill="y", pady=18)

        # Shortcut hints
        hint_frame = tk.Frame(right_cluster, bg=C["panel"])
        hint_frame.pack(side="left", padx=(0, 30))
        for key, label in [("Ctrl+O", "Open"), ("Ctrl+X", "Clear")]:
            row = tk.Frame(hint_frame, bg=C["panel"])
            row.pack(anchor="e")
            tk.Label(row, text=key, bg=C["border"], fg=C["teal"],
                     font=("Courier New", 8, "bold"),
                     padx=5, pady=1, relief="flat").pack(side="left", padx=(0, 5))
            tk.Label(row, text=label, bg=C["panel"], fg=C["text_dim"],
                     font=("Courier New", 8)).pack(side="left")

        # Status pill
        status_frame = tk.Frame(right_cluster,
                                bg=C["teal_dark"] if self.model else C["crimson_dim"],
                                padx=14, pady=6)
        status_frame.pack(side="left")

        self._status_dot = PulseDot(
            status_frame,
            color=C["teal"] if self.model else C["crimson"])
        self._status_dot.pack(side="left", padx=(0, 8))

        self._status_lbl = tk.Label(
            status_frame,
            text="MODEL READY" if self.model else "MODEL OFFLINE",
            bg=C["teal_dark"] if self.model else C["crimson_dim"],
            fg=C["teal"] if self.model else C["crimson"],
            font=("Courier New", 10, "bold"))
        self._status_lbl.pack(side="left")

    def _draw_cell_logo(self, canvas):
        """Draw a stylised blood cell icon on canvas."""
        w, h = 40, 40
        cx, cy = 20, 20
        # Outer ring
        canvas.create_oval(3, 3, 37, 37, outline=C["crimson"], width=2)
        # Inner dimple
        canvas.create_oval(12, 12, 28, 28, outline=C["crimson_dim"], width=1,
                           dash=(3, 2))
        # Cross hatch
        canvas.create_line(20, 4, 20, 36, fill=C["crimson_dim"], width=1, dash=(2, 3))
        canvas.create_line(4, 20, 36, 20, fill=C["crimson_dim"], width=1, dash=(2, 3))

    # ── Left Panel ────────────────────────────────────────────────────────────
    def _build_left(self, parent):
        parent.rowconfigure(1, weight=1)
        parent.columnconfigure(0, weight=1)

        # Section header
        sec_hdr = tk.Frame(parent, bg=C["card"],
                           highlightbackground=C["border"], highlightthickness=1)
        sec_hdr.grid(row=0, column=0, sticky="ew")
        tk.Label(sec_hdr, text="⬡  SPECIMEN INPUT", bg=C["card"], fg=C["teal"],
                 font=("Courier New", 11, "bold"), padx=20, pady=12).pack(side="left")

        # Drop zone
        dz_outer = tk.Frame(parent, bg=C["panel"])
        dz_outer.grid(row=1, column=0, sticky="nsew", padx=20, pady=16)
        dz_outer.rowconfigure(0, weight=1)
        dz_outer.columnconfigure(0, weight=1)

        self.drop_zone = tk.Canvas(
            dz_outer, bg=C["bg"],
            highlightbackground=C["teal_dim"], highlightthickness=1,
            cursor="hand2")
        self.drop_zone.grid(row=0, column=0, sticky="nsew")
        self.drop_zone.bind("<Configure>", self._render_drop_placeholder)
        self.drop_zone.bind("<Button-1>", lambda e: self.upload_image())

        # Image info strip
        self.img_info_var = tk.StringVar(value="")
        info_bar = tk.Frame(parent, bg=C["card"])
        info_bar.grid(row=2, column=0, sticky="ew")
        self._img_info_lbl = tk.Label(
            info_bar, textvariable=self.img_info_var,
            bg=C["card"], fg=C["text_mid"],
            font=("Courier New", 9), padx=16, pady=6, anchor="w")
        self._img_info_lbl.pack(fill="x")

        # Button bar
        btn_bar = tk.Frame(parent, bg=C["panel"])
        btn_bar.grid(row=3, column=0, sticky="ew", padx=20, pady=(8, 18))
        btn_bar.columnconfigure(0, weight=1)
        btn_bar.columnconfigure(1, weight=1)

        self._upload_btn = self._make_btn(btn_bar, "▲  SELECT IMAGE",
                                          self.upload_image,
                                          bg=C["teal"], fg=C["bg"],
                                          active_bg=C["teal_dim"])
        self._upload_btn.grid(row=0, column=0, sticky="ew", padx=(0, 8))

        self._clear_btn = self._make_btn(btn_bar, "✕  CLEAR",
                                         self.clear_image,
                                         bg=C["card"], fg=C["text_mid"],
                                         active_bg=C["border"],
                                         border=C["border"])
        self._clear_btn.grid(row=0, column=1, sticky="ew")

    def _render_drop_placeholder(self, event=None):
        c = self.drop_zone
        c.delete("placeholder")
        if self.current_image_path:
            return
        w = c.winfo_width()
        h = c.winfo_height()
        if w < 2 or h < 2:
            return
        cx, cy = w // 2, h // 2
        # Dashed border rectangle
        margin = 24
        for i in range(0, w - margin * 2, 12):
            c.create_line(margin + i, margin, margin + i + 7, margin,
                          fill=C["border_glow"], width=1, tags="placeholder")
            c.create_line(margin + i, h - margin, margin + i + 7, h - margin,
                          fill=C["border_glow"], width=1, tags="placeholder")
        for i in range(0, h - margin * 2, 12):
            c.create_line(margin, margin + i, margin, margin + i + 7,
                          fill=C["border_glow"], width=1, tags="placeholder")
            c.create_line(w - margin, margin + i, w - margin, margin + i + 7,
                          fill=C["border_glow"], width=1, tags="placeholder")

        # Center icon
        icon_r = 36
        c.create_oval(cx - icon_r, cy - icon_r - 20,
                      cx + icon_r, cy + icon_r - 20,
                      outline=C["teal_dim"], width=1.5,
                      dash=(4, 3), tags="placeholder")
        c.create_text(cx, cy - 20, text="⊕", fill=C["teal_dim"],
                      font=("Courier New", 28), tags="placeholder")
        c.create_text(cx, cy + 28,
                      text="Click or drag a blood cell image",
                      fill=C["text_mid"], font=("Courier New", 11),
                      tags="placeholder")
        c.create_text(cx, cy + 48,
                      text="JPG  ·  PNG  ·  BMP  ·  TIFF",
                      fill=C["text_dim"], font=("Courier New", 9),
                      tags="placeholder")

    # ── Right Panel ───────────────────────────────────────────────────────────
    def _build_right(self, parent):
        parent.rowconfigure(2, weight=1)
        parent.columnconfigure(0, weight=1)

        # Section header
        sec_hdr = tk.Frame(parent, bg=C["card"],
                           highlightbackground=C["border"], highlightthickness=1)
        sec_hdr.grid(row=0, column=0, sticky="ew")
        tk.Label(sec_hdr, text="⬡  ANALYSIS OUTPUT", bg=C["card"], fg=C["violet"],
                 font=("Courier New", 11, "bold"), padx=20, pady=12).pack(side="left")

        # ─ Result Hero Card ─
        hero = tk.Frame(parent, bg=C["card"],
                        highlightbackground=C["border"], highlightthickness=1)
        hero.grid(row=1, column=0, sticky="ew", padx=20, pady=(18, 0))
        hero.columnconfigure(0, weight=1)

        # Scan-line decoration (thin teal line across top of hero)
        scan_accent = tk.Frame(hero, bg=C["teal"], height=2)
        scan_accent.pack(fill="x", side="top")

        hero_inner = tk.Frame(hero, bg=C["card"])
        hero_inner.pack(fill="x", padx=22, pady=18)
        hero_inner.columnconfigure(1, weight=1)

        # Big hex badge
        self._hero_badge = HexBadge(hero_inner, color=C["text_dim"],
                                    icon="?", size=64)
        self._hero_badge.grid(row=0, column=0, rowspan=2,
                              padx=(0, 20), sticky="ns")

        # Prediction text
        self._pred_lbl = tk.Label(
            hero_inner, text="AWAITING BLOODCELL IMAGE",
            bg=C["card"], fg=C["text_dim"],
            font=("Courier New", 20, "bold"), anchor="w")
        self._pred_lbl.grid(row=0, column=1, sticky="ew")

        self._pred_desc = tk.Label(
            hero_inner, text="Upload an image to begin classification",
            bg=C["card"], fg=C["text_dim"],
            font=("Courier New", 9), anchor="w")
        self._pred_desc.grid(row=1, column=1, sticky="ew")

        # Confidence meter
        conf_frame = tk.Frame(hero, bg=C["card"])
        conf_frame.pack(fill="x", padx=22, pady=(0, 8))

        conf_header = tk.Frame(conf_frame, bg=C["card"])
        conf_header.pack(fill="x", pady=(0, 6))
        tk.Label(conf_header, text="CONFIDENCE", bg=C["card"],
                 fg=C["text_dim"], font=("Courier New", 8, "bold")).pack(side="left")
        self._conf_pct = tk.Label(conf_header, text="—",
                                  bg=C["card"], fg=C["teal"],
                                  font=("Courier New", 8, "bold"))
        self._conf_pct.pack(side="right")

        self._conf_bar = GradientBar(
            conf_frame, height=8,
            bg_color=C["border"],
            accent_start=C["teal"], accent_end=C["green"])
        self._conf_bar.pack(fill="x")
        self._conf_bar.set(0, animate=False)

        # ─ Class Probabilities ─
        probs_hdr = tk.Frame(parent, bg=C["panel"])
        probs_hdr.grid(row=2, column=0, sticky="ew", padx=20, pady=(18, 0))
        tk.Label(probs_hdr, text="CLASS PROBABILITIES",
                 bg=C["panel"], fg=C["text_mid"],
                 font=("Courier New", 9, "bold")).pack(side="left")

        # Scrollable class list
        class_outer = tk.Frame(parent, bg=C["panel"])
        class_outer.grid(row=3, column=0, sticky="nsew",
                         padx=20, pady=(10, 0))
        parent.rowconfigure(3, weight=1)

        self._class_rows = {}
        for cls in self.class_names:
            row = self._make_class_row(class_outer, cls)
            row.pack(fill="x", pady=6)
            self._class_rows[cls] = row

        # ─ Meta strip ─
        meta_strip = tk.Frame(parent, bg=C["card"],
                              highlightbackground=C["border"],
                              highlightthickness=1)
        meta_strip.grid(row=4, column=0, sticky="ew", padx=20, pady=18)
        meta_strip.columnconfigure(0, weight=1)
        meta_strip.columnconfigure(1, weight=1)
        meta_strip.columnconfigure(2, weight=1)

        for i, (icon, label, attr) in enumerate([
            ("⚙", "DEVICE", self.device.upper() if hasattr(self.device, 'upper') else str(self.device).upper()),
            ("◈", "CLASSES", str(CONFIG["num_classes"])),
            ("▣", "IMG SIZE", f"{CONFIG['img_size']}×{CONFIG['img_size']}"),
        ]):
            cell = tk.Frame(meta_strip, bg=C["card"])
            cell.grid(row=0, column=i, sticky="ew", padx=1)
            tk.Label(cell, text=f"{icon}  {label}",
                     bg=C["card"], fg=C["text_dim"],
                     font=("Courier New", 7, "bold")).pack(pady=(10, 2))
            tk.Label(cell, text=attr,
                     bg=C["card"], fg=C["teal"],
                     font=("Courier New", 11, "bold")).pack(pady=(0, 10))

    def _make_class_row(self, parent, cls_name):
        color = CLASS_COLORS.get(cls_name, C["teal"])
        icon = CLASS_ICONS.get(cls_name, "◆")
        desc = CLASS_DESC.get(cls_name, "")

        container = tk.Frame(parent, bg=C["card"],
                             highlightbackground=C["border"],
                             highlightthickness=1)

        inner = tk.Frame(container, bg=C["card"])
        inner.pack(fill="x", padx=14, pady=10)
        inner.columnconfigure(1, weight=1)

        # Hex mini badge
        badge = HexBadge(inner, color=color, icon=icon, size=36)
        badge.grid(row=0, column=0, rowspan=2, padx=(0, 12), sticky="ns")

        # Name + description
        tk.Label(inner, text=cls_name.upper(),
                 bg=C["card"], fg=color,
                 font=("Courier New", 11, "bold"),
                 anchor="w").grid(row=0, column=1, sticky="ew")
        tk.Label(inner, text=desc,
                 bg=C["card"], fg=C["text_dim"],
                 font=("Courier New", 8),
                 anchor="w").grid(row=1, column=1, sticky="ew")

        # Percentage label
        pct_var = tk.StringVar(value="0.0%")
        pct_lbl = tk.Label(inner, textvariable=pct_var,
                           bg=C["card"], fg=C["text_mid"],
                           font=("Courier New", 12, "bold"), width=6, anchor="e")
        pct_lbl.grid(row=0, column=2, rowspan=2, padx=(8, 0), sticky="ns")

        # Gradient bar
        bar = GradientBar(container, height=5,
                          bg_color=C["border"],
                          accent_start=color, accent_end=C["white"])
        bar.pack(fill="x", padx=14, pady=(0, 10))
        bar.set(0, animate=False)

        # Store references
        container._cls_bar = bar
        container._cls_pct = pct_var
        container._cls_lbl = pct_lbl
        container._cls_color = color
        container._cls_badge = badge
        return container

    # ── Footer ────────────────────────────────────────────────────────────────
    def _build_footer(self):
        foot = tk.Frame(self.root, bg=C["panel"],
                        highlightbackground=C["border"], highlightthickness=1)
        foot.pack(fill="x", side="bottom")

        # Teal accent line at top of footer
        tk.Frame(foot, bg=C["teal_dark"], height=1).pack(fill="x")

        inner = tk.Frame(foot, bg=C["panel"])
        inner.pack(fill="x", padx=28, pady=8)

        tk.Label(inner,
                 text="🔬  HemaScan AI  ·  For Research Use Only  ·  Not for Clinical Diagnosis",
                 bg=C["panel"], fg=C["text_dim"],
                 font=("Courier New", 9)).pack(side="left")

        # Live clock
        self._clock_var = tk.StringVar()
        tk.Label(inner, textvariable=self._clock_var,
                 bg=C["panel"], fg=C["text_dim"],
                 font=("Courier New", 9)).pack(side="right")
        self._tick_clock()

    def _tick_clock(self):
        import datetime
        self._clock_var.set(
            datetime.datetime.now().strftime("SYS  %Y-%m-%d  %H:%M:%S"))
        self.root.after(1000, self._tick_clock)

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _make_btn(self, parent, text, cmd, bg, fg,
                  active_bg=None, border=None):
        btn = tk.Button(parent, text=text, command=cmd,
                        bg=bg, fg=fg, activebackground=active_bg or bg,
                        activeforeground=fg, relief="flat", cursor="hand2",
                        font=("Courier New", 10, "bold"), pady=10, padx=14,
                        highlightthickness=1 if border else 0,
                        highlightbackground=border or bg,
                        bd=0)
        btn.bind("<Enter>", lambda e: btn.configure(bg=active_bg or bg))
        btn.bind("<Leave>", lambda e: btn.configure(bg=bg))
        return btn

    # ── Shortcuts ─────────────────────────────────────────────────────────────
    def setup_shortcuts(self):
        self.root.bind('<Control-o>', lambda e: self.upload_image())
        self.root.bind('<Control-x>', lambda e: self.clear_image())

    # ── Core Logic ────────────────────────────────────────────────────────────
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Blood Cell Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path)
            self.predict_image(file_path)

    def display_image(self, image_path):
        try:
            img = Image.open(image_path).convert("RGB")
            self.current_image = img

            c = self.drop_zone
            c.delete("all")

            cw = c.winfo_width() or 400
            ch = c.winfo_height() or 400
            display_size = (cw - 8, ch - 8)
            thumb = img.copy()
            thumb.thumbnail(display_size, Image.Resampling.LANCZOS)

            photo = ImageTk.PhotoImage(thumb)
            c.create_image(cw // 2, ch // 2, image=photo, anchor="center")
            c.image = photo  # prevent GC

            # Overlay — corner brackets for a "scanning" feel
            blen = 20
            bthk = 2
            for x, y, dx, dy in [
                (4, 4, 1, 1), (cw - 4, 4, -1, 1),
                (4, ch - 4, 1, -1), (cw - 4, ch - 4, -1, -1)
            ]:
                c.create_line(x, y, x + dx * blen, y,
                              fill=C["teal"], width=bthk)
                c.create_line(x, y, x, y + dy * blen,
                              fill=C["teal"], width=bthk)

            file_size = os.path.getsize(image_path) / 1024
            orig_w, orig_h = img.size
            self.img_info_var.set(
                f"  ▣  {orig_w} × {orig_h} px    ▣  {file_size:.1f} KB"
                f"    ▣  {os.path.basename(image_path)}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to display image: {e}")

    def predict_image(self, image_path):
        if not self.model:
            messagebox.showerror("Error",
                "Model not loaded. Please train the model first.")
            return
        try:
            img = Image.open(image_path).convert("RGB")
            tensor = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(tensor)
                probs = torch.softmax(logits, dim=1)[0]
                pred_idx = probs.argmax().item()
                confidence = probs[pred_idx].item()
            predicted_class = self.class_names[pred_idx]
            self.update_results(predicted_class, confidence,
                                probs.cpu().numpy())
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {e}")

    def update_results(self, predicted_class, confidence, probabilities):
        color = CLASS_COLORS.get(predicted_class, C["teal"])
        icon = CLASS_ICONS.get(predicted_class, "◆")
        desc = CLASS_DESC.get(predicted_class, "")

        # Hero badge
        self._hero_badge.config(bg=C["card"])
        self._hero_badge.color = color
        self._hero_badge.icon = icon
        self._hero_badge._draw()

        # Prediction text
        self._pred_lbl.configure(text=predicted_class.upper(), fg=color)
        self._pred_desc.configure(text=desc, fg=C["text_mid"])

        # Confidence
        conf_pct = confidence * 100
        self._conf_pct.configure(
            text=f"{conf_pct:.1f}%",
            fg=C["green"] if conf_pct >= 80 else C["amber"] if conf_pct >= 55 else C["crimson"])

        self._conf_bar.accent_start = color
        self._conf_bar.set(confidence)

        # Class rows
        for i, cls in enumerate(self.class_names):
            prob = float(probabilities[i])
            row = self._class_rows[cls]
            row._cls_bar.accent_start = CLASS_COLORS.get(cls, C["teal"])
            row._cls_bar.set(prob)
            row._cls_pct.set(f"{prob * 100:.1f}%")

            if cls == predicted_class:
                row._cls_lbl.configure(fg=CLASS_COLORS.get(cls, C["teal"]))
                row.configure(
                    highlightbackground=CLASS_COLORS.get(cls, C["teal"]))
            else:
                row._cls_lbl.configure(fg=C["text_mid"])
                row.configure(highlightbackground=C["border"])

        # Glow flash on hero card
        self._flash_hero(color)

    def _flash_hero(self, color):
        """Brief accent flash on the hero card."""
        try:
            hero_children = self.root.winfo_children()
            # Quick border pulse via status dot color shift
            self._status_dot.set_color(color)
            self.root.after(800, lambda: self._status_dot.set_color(
                C["teal"] if self.model else C["crimson"]))
        except Exception:
            pass

    def clear_image(self):
        self.current_image_path = None
        self.current_image = None

        # Reset drop zone
        self.drop_zone.delete("all")
        self._render_drop_placeholder()
        self.img_info_var.set("")

        # Reset hero
        self._hero_badge.color = C["text_dim"]
        self._hero_badge.icon = "?"
        self._hero_badge._draw()
        self._pred_lbl.configure(text="AWAITING SPECIMEN", fg=C["text_dim"])
        self._pred_desc.configure(
            text="Upload an image to begin classification",
            fg=C["text_dim"])
        self._conf_pct.configure(text="—", fg=C["teal"])
        self._conf_bar.set(0)

        # Reset class rows
        for cls, row in self._class_rows.items():
            row._cls_bar.set(0)
            row._cls_pct.set("0.0%")
            row._cls_lbl.configure(fg=C["text_mid"])
            row.configure(highlightbackground=C["border"])

    def run(self):
        self.root.mainloop()


def main():
    print("=" * 55)
    print("  HemaScan AI — Blood Cell Classifier")
    print("=" * 55)
    app = BloodCellClassifierGUI()
    app.run()


if __name__ == "__main__":
    main()
