"""
MangaTR-Studio - Desktop Localization Studio GUI (app/ui/main_window.py)
"""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

from app.config import FONTS_DIR, INPUT_DIR, OUTPUT_DIR, StudioConfig
from app.core.batch_processor import BatchProcessor
from app.core.pipeline import MangaPipeline, PagePipelineResult


class MangaTRMainWindow:
    """
    Modern dark-themed Desktop GUI for MangaTR-Studio:
    - Single page interactive OCR + Translation + Cleaning + Typesetting
    - Live bubble text & font editor with instant re-typesetting
    - Side-by-side comparison mode (Original vs. Turkish)
    - Series Glossary / Terminology manager
    - Multi-chapter (400+ chapters) Batch Processor with progress bar
    """

    BG_DARK = "#14171f"
    BG_PANEL = "#1d222e"
    BG_CARD = "#252b3b"
    FG_MAIN = "#f0f4f8"
    FG_MUTED = "#9aa5b8"
    ACCENT = "#3b82f6"
    ACCENT_SUCCESS = "#10b981"

    def __init__(self, root: tk.Tk, pipeline: Optional[MangaPipeline] = None) -> None:
        self.root = root
        self.root.title("MangaTR Studio — AI Manga Çeviri & Typesetting Stüdyosu")
        self.root.geometry("1460x900")
        self.root.minsize(1120, 720)
        self.root.configure(bg=self.BG_DARK)

        self.config = StudioConfig()
        self.pipeline = pipeline or MangaPipeline(self.config)
        self.batch_processor = BatchProcessor(self.pipeline)

        self.current_image_path: Optional[Path] = None
        self.current_result: Optional[PagePipelineResult] = None
        self.selected_block_idx: Optional[int] = None

        self.view_mode = tk.StringVar(value="final_tr")
        self.backend_var = tk.StringVar(value=self.config.translator.backend)
        self.reading_order_var = tk.StringVar(value=self.config.detector.reading_order)
        self.status_var = tk.StringVar(
            value=f"Hazır — Cihaz: {self.pipeline.device.upper()} | Proje: {self.pipeline.project.project_name}"
        )
        self.progress_var = tk.DoubleVar(value=0.0)

        self.zoom_scale: float = 1.0
        self._tk_image: Optional[ImageTk.PhotoImage] = None
        self._is_busy: bool = False

        self._setup_styles()
        self._build_layout()

        # Automatically load input/test.jpg if available so the user sees it immediately
        default_test = INPUT_DIR / "test.jpg"
        if default_test.exists():
            self.load_image_file(default_test)
            existing_out = OUTPUT_DIR / "test_tr.jpg"
            existing_json = OUTPUT_DIR / "test_tr.json"
            if existing_out.exists() and existing_json.exists():
                self._try_load_cached_result(default_test)

    def _setup_styles(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure(".", background=self.BG_PANEL, foreground=self.FG_MAIN, font=("Segoe UI", 10))
        style.configure("TFrame", background=self.BG_PANEL)
        style.configure("Dark.TFrame", background=self.BG_DARK)
        style.configure("Card.TFrame", background=self.BG_CARD)
        style.configure("TLabel", background=self.BG_PANEL, foreground=self.FG_MAIN)
        style.configure("Muted.TLabel", background=self.BG_PANEL, foreground=self.FG_MUTED, font=("Segoe UI", 9))
        style.configure("Header.TLabel", background=self.BG_PANEL, foreground=self.FG_MAIN, font=("Segoe UI Semibold", 11))
        style.configure(
            "Accent.TButton",
            background=self.ACCENT,
            foreground="#ffffff",
            font=("Segoe UI Semibold", 10),
            padding=(12, 7),
        )
        style.map("Accent.TButton", background=[("active", "#2563eb")])
        style.configure(
            "Success.TButton",
            background=self.ACCENT_SUCCESS,
            foreground="#ffffff",
            font=("Segoe UI Semibold", 10),
            padding=(12, 7),
        )
        style.map("Success.TButton", background=[("active", "#059669")])
        style.configure("TButton", background=self.BG_CARD, foreground=self.FG_MAIN, padding=(10, 6))
        style.map("TButton", background=[("active", "#333c52")])

    def _build_layout(self) -> None:
        # 1. Top Action Bar
        top_bar = ttk.Frame(self.root, padding=(12, 8))
        top_bar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(top_bar, text="📂 Manga Sayfası Aç", command=self.on_open_image).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(
            top_bar,
            text="⚡ Sayfayı Çevir & Yerleştir",
            style="Accent.TButton",
            command=self.on_translate_current_page,
        ).pack(side=tk.LEFT, padx=6)
        ttk.Button(
            top_bar,
            text="📁 Toplu Bölüm Çevir (Batch)",
            style="Success.TButton",
            command=self.on_open_batch_dialog,
        ).pack(side=tk.LEFT, padx=6)
        ttk.Button(top_bar, text="📖 Terimler Sözlüğü", command=self.on_open_glossary_dialog).pack(side=tk.LEFT, padx=6)
        ttk.Button(top_bar, text="🔑 API Ayarları", command=self.on_open_api_settings_dialog).pack(side=tk.LEFT, padx=6)
        ttk.Button(top_bar, text="💾 Farklı Kaydet", command=self.on_export_image).pack(side=tk.LEFT, padx=6)

        # Right side of top bar: Translation Engine & Reading Order
        ttk.Label(top_bar, text="Okuma Yönü:", style="Muted.TLabel").pack(side=tk.RIGHT, padx=(8, 4))
        order_cb = ttk.Combobox(
            top_bar,
            textvariable=self.reading_order_var,
            values=["manga_rtl", "comic_ltr"],
            width=11,
            state="readonly",
        )
        order_cb.pack(side=tk.RIGHT, padx=(0, 10))

        ttk.Label(top_bar, text="Çeviri Motoru:", style="Muted.TLabel").pack(side=tk.RIGHT, padx=(8, 4))
        backend_cb = ttk.Combobox(
            top_bar,
            textvariable=self.backend_var,
            values=["auto", "free_web", "local_llm", "gemini_api", "openai_api"],
            width=12,
            state="readonly",
        )
        backend_cb.pack(side=tk.RIGHT)

        # 2. Main Split Area (Center Canvas + Right Dialogue Inspector)
        main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(4, 6))

        # Center Viewer Panel
        viewer_frame = ttk.Frame(main_pane, padding=6)
        main_pane.add(viewer_frame, weight=3)

        # View Mode Selector Row
        mode_bar = ttk.Frame(viewer_frame)
        mode_bar.pack(side=tk.TOP, fill=tk.X, pady=(0, 6))

        modes = [
            ("🇹🇷 Türkçe Final", "final_tr"),
            ("🔍 Yan Yana (Orijinal | Türkçe)", "side_by_side"),
            ("🧹 Temizlenmiş (Inpaint)", "cleaned"),
            ("🎯 Tespit Kutuları (BBox)", "detected"),
            ("📄 Orijinal", "original"),
        ]
        for label, val in modes:
            rb = ttk.Radiobutton(
                mode_bar,
                text=label,
                value=val,
                variable=self.view_mode,
                command=self._refresh_canvas_image,
            )
            rb.pack(side=tk.LEFT, padx=6)

        ttk.Button(mode_bar, text="🔍 Ekrana Sığdır", command=self.on_fit_zoom).pack(side=tk.RIGHT, padx=4)
        ttk.Button(mode_bar, text="➕", width=3, command=lambda: self.on_zoom(1.2)).pack(side=tk.RIGHT, padx=2)
        ttk.Button(mode_bar, text="➖", width=3, command=lambda: self.on_zoom(0.83)).pack(side=tk.RIGHT, padx=2)

        # Scrollable Image Canvas
        canvas_container = ttk.Frame(viewer_frame, style="Dark.TFrame")
        canvas_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(
            canvas_container,
            bg="#0e1017",
            highlightthickness=0,
            cursor="crosshair",
        )
        v_scroll = ttk.Scrollbar(canvas_container, orient=tk.VERTICAL, command=self.canvas.yview)
        h_scroll = ttk.Scrollbar(canvas_container, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)

        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas.bind("<Configure>", lambda e: self._refresh_canvas_image())
        self.canvas.bind("<ButtonPress-1>", self._on_canvas_click)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)

        # Right Dialogue Editor Panel
        right_panel = ttk.Frame(main_pane, padding=10)
        main_pane.add(right_panel, weight=1)

        ttk.Label(right_panel, text="Konuşma Balonları (Okuma Sırası)", style="Header.TLabel").pack(
            anchor=tk.W, pady=(0, 6)
        )

        # Treeview of blocks
        columns = ("id", "en", "tr")
        self.block_tree = ttk.Treeview(right_panel, columns=columns, show="headings", height=12, selectmode="browse")
        self.block_tree.heading("id", text="#")
        self.block_tree.heading("en", text="İngilizce (OCR)")
        self.block_tree.heading("tr", text="Türkçe Çeviri")
        self.block_tree.column("id", width=38, anchor=tk.CENTER, stretch=False)
        self.block_tree.column("en", width=175, anchor=tk.W)
        self.block_tree.column("tr", width=195, anchor=tk.W)
        self.block_tree.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.block_tree.bind("<<TreeviewSelect>>", self.on_tree_select)

        # Selected Block Editor Card
        editor_frame = ttk.Frame(right_panel, padding=8, style="Card.TFrame")
        editor_frame.pack(fill=tk.X)

        self.selected_label = ttk.Label(editor_frame, text="Seçili Balon: Yok", style="Header.TLabel")
        self.selected_label.pack(anchor=tk.W, pady=(0, 6))

        ttk.Label(editor_frame, text="İngilizce Metin (OCR):", style="Muted.TLabel").pack(anchor=tk.W)
        self.en_text_widget = tk.Text(
            editor_frame,
            height=3,
            bg="#141824",
            fg=self.FG_MAIN,
            insertbackground=self.FG_MAIN,
            font=("Segoe UI", 10),
            wrap=tk.WORD,
            relief=tk.FLAT,
        )
        self.en_text_widget.pack(fill=tk.X, pady=(2, 8))

        ttk.Label(editor_frame, text="Türkçe Çeviri (Balona Basılacak Metin):", style="Muted.TLabel").pack(anchor=tk.W)
        self.tr_text_widget = tk.Text(
            editor_frame,
            height=4,
            bg="#141824",
            fg="#60a5fa",
            insertbackground=self.FG_MAIN,
            font=("Segoe UI Semibold", 10),
            wrap=tk.WORD,
            relief=tk.FLAT,
        )
        self.tr_text_widget.pack(fill=tk.X, pady=(2, 8))

        # Font & Size Controls
        font_row = ttk.Frame(editor_frame)
        font_row.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(font_row, text="Font:", style="Muted.TLabel").pack(side=tk.LEFT)
        available_fonts = [""] + [p.name for p in sorted(FONTS_DIR.glob("*.ttf"))]
        self.block_font_var = tk.StringVar(value="")
        self.font_cb = ttk.Combobox(font_row, textvariable=self.block_font_var, values=available_fonts, width=20)
        self.font_cb.pack(side=tk.LEFT, padx=(4, 10))

        ttk.Label(font_row, text="Boyut (0=Oto):", style="Muted.TLabel").pack(side=tk.LEFT)
        self.block_size_var = tk.IntVar(value=0)
        size_spin = ttk.Spinbox(font_row, from_=0, to=48, textvariable=self.block_size_var, width=5)
        size_spin.pack(side=tk.LEFT, padx=4)

        btn_row = ttk.Frame(editor_frame)
        btn_row.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(
            btn_row,
            text="🌍 Bu Balonu Tekrar Çevir",
            command=self.on_retranslate_selected_block,
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))
        ttk.Button(
            btn_row,
            text="✍️ Uygula & Yeniden Bas",
            style="Accent.TButton",
            command=self.on_apply_block_edit,
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(4, 0))

        # 3. Bottom Status & Progress Bar
        bottom_bar = ttk.Frame(self.root, padding=(12, 6))
        bottom_bar.pack(side=tk.BOTTOM, fill=tk.X)

        ttk.Label(bottom_bar, textvariable=self.status_var).pack(side=tk.LEFT)
        self.progress_bar = ttk.Progressbar(
            bottom_bar,
            variable=self.progress_var,
            maximum=100.0,
            length=260,
            mode="determinate",
        )
        self.progress_bar.pack(side=tk.RIGHT, padx=6)

    # ------------------------------------------------------------------
    # Image Loading & Rendering
    # ------------------------------------------------------------------

    def load_image_file(self, path: Path) -> None:
        self.current_image_path = path.resolve()
        self.current_result = None
        self.selected_block_idx = None
        self.block_tree.delete(*self.block_tree.get_children())

        img_bytes = np.fromfile(str(self.current_image_path), dtype=np.uint8)
        bgr = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        if bgr is None:
            messagebox.showerror("Hata", f"Görsel okunamadı:\n{path}")
            return

        self._raw_loaded_bgr = bgr
        self.view_mode.set("original")
        self.zoom_scale = 1.0
        self.status_var.set(f"Yüklendi: {path.name} ({bgr.shape[1]}x{bgr.shape[0]})")
        self.root.after(50, self.on_fit_zoom)

    def _try_load_cached_result(self, src_path: Path) -> None:
        """If output/test_tr.jpg and output/test_tr.json already exist, load them for instant preview."""
        from app.utils.json_utils import load_json

        stem = src_path.stem
        tr_path = OUTPUT_DIR / f"{stem}_tr.jpg"
        clean_path = OUTPUT_DIR / f"{stem}_cleaned.jpg"
        det_path = OUTPUT_DIR / f"{stem}_detected.jpg"
        json_path = OUTPUT_DIR / f"{stem}_tr.json"
        if not (tr_path.exists() and json_path.exists()):
            return

        try:
            meta = load_json(json_path)
            tr_bgr = cv2.imdecode(np.fromfile(str(tr_path), dtype=np.uint8), cv2.IMREAD_COLOR)
            clean_bgr = (
                cv2.imdecode(np.fromfile(str(clean_path), dtype=np.uint8), cv2.IMREAD_COLOR)
                if clean_path.exists()
                else self._raw_loaded_bgr.copy()
            )
            det_bgr = (
                cv2.imdecode(np.fromfile(str(det_path), dtype=np.uint8), cv2.IMREAD_COLOR)
                if det_path.exists()
                else self._raw_loaded_bgr.copy()
            )
            blocks = meta.get("blocks", [])
            zeros = np.zeros(self._raw_loaded_bgr.shape[:2], dtype=np.uint8)
            self.current_result = PagePipelineResult(
                input_path=src_path,
                original_bgr=self._raw_loaded_bgr,
                detected_vis_bgr=det_bgr,
                raw_mask=zeros,
                refined_mask=zeros,
                cleaned_bgr=clean_bgr,
                final_tr_bgr=tr_bgr,
                blocks=blocks,
                clean_infos=[],
                output_image_path=tr_path,
                output_json_path=json_path,
            )
            self._populate_block_tree()
            self.view_mode.set("final_tr")
            self._refresh_canvas_image()
            self.status_var.set(f"Hazır: {src_path.name} ({len(blocks)} konuşma bloğu yüklendi)")
        except Exception:
            pass

    def _get_active_display_bgr(self) -> Optional[np.ndarray]:
        mode = self.view_mode.get()
        if self.current_result is None:
            return getattr(self, "_raw_loaded_bgr", None)

        if mode == "original":
            base = self.current_result.original_bgr.copy()
        elif mode == "detected":
            base = self.current_result.detected_vis_bgr.copy()
        elif mode == "cleaned":
            base = self.current_result.cleaned_bgr.copy()
        elif mode == "side_by_side":
            left = self.current_result.original_bgr
            right = self.current_result.final_tr_bgr
            divider = np.full((left.shape[0], 6, 3), (59, 130, 246), dtype=np.uint8)
            return np.hstack([left, divider, right])
        else:
            base = self.current_result.final_tr_bgr.copy()

        # Highlight currently selected block
        if self.selected_block_idx is not None and 0 <= self.selected_block_idx < len(self.current_result.blocks):
            blk = self.current_result.blocks[self.selected_block_idx]
            x1, y1, x2, y2 = blk["bbox"]
            cv2.rectangle(base, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (246, 130, 59), 3)

        return base

    def _refresh_canvas_image(self) -> None:
        bgr = self._get_active_display_bgr()
        if bgr is None:
            return

        h, w = bgr.shape[:2]
        new_w = max(50, int(round(w * self.zoom_scale)))
        new_h = max(50, int(round(h * self.zoom_scale)))

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb).resize((new_w, new_h), Image.Resampling.LANCZOS)
        self._tk_image = ImageTk.PhotoImage(pil_img)

        self.canvas.delete("all")
        cw = max(new_w, self.canvas.winfo_width())
        ch = max(new_h, self.canvas.winfo_height())
        ox = max(0, (cw - new_w) // 2)
        oy = max(0, (ch - new_h) // 2)
        self._canvas_img_offset = (ox, oy)
        self.canvas.create_image(ox, oy, anchor=tk.NW, image=self._tk_image)
        self.canvas.configure(scrollregion=(0, 0, cw, ch))

    def on_fit_zoom(self) -> None:
        bgr = self._get_active_display_bgr()
        if bgr is None:
            return
        cw = max(200, self.canvas.winfo_width() - 24)
        ch = max(200, self.canvas.winfo_height() - 24)
        h, w = bgr.shape[:2]
        self.zoom_scale = min(cw / float(w), ch / float(h), 1.25)
        self._refresh_canvas_image()

    def on_zoom(self, factor: float) -> None:
        self.zoom_scale = max(0.2, min(3.5, self.zoom_scale * factor))
        self._refresh_canvas_image()

    def _on_mouse_wheel(self, event: tk.Event) -> None:
        if event.delta > 0:
            self.on_zoom(1.12)
        else:
            self.on_zoom(0.89)

    def _on_canvas_click(self, event: tk.Event) -> None:
        self.canvas.scan_mark(event.x, event.y)
        if self.current_result is None or self.view_mode.get() == "side_by_side":
            return

        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        ox, oy = getattr(self, "_canvas_img_offset", (0, 0))
        img_x = (cx - ox) / max(0.01, self.zoom_scale)
        img_y = (cy - oy) / max(0.01, self.zoom_scale)

        for idx, blk in enumerate(self.current_result.blocks):
            x1, y1, x2, y2 = blk["bbox"]
            if x1 <= img_x <= x2 and y1 <= img_y <= y2:
                children = self.block_tree.get_children()
                if idx < len(children):
                    self.block_tree.selection_set(children[idx])
                    self.block_tree.see(children[idx])
                    self._select_block_by_index(idx)
                break

    def _on_canvas_drag(self, event: tk.Event) -> None:
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    # ------------------------------------------------------------------
    # Actions & Callbacks
    # ------------------------------------------------------------------

    def on_open_image(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Manga Sayfası Seç",
            initialdir=str(INPUT_DIR),
            filetypes=[("Manga Görselleri", "*.jpg *.jpeg *.png *.webp *.bmp")],
        )
        if file_path:
            self.load_image_file(Path(file_path))

    def on_translate_current_page(self) -> None:
        if self._is_busy:
            return
        if self.current_image_path is None:
            messagebox.showwarning("Uyarı", "Lütfen önce bir manga görseli açın.")
            return

        self.config.translator.backend = self.backend_var.get()
        self.config.detector.reading_order = self.reading_order_var.get()
        self._is_busy = True

        def worker() -> None:
            try:
                def prog(msg: str, pct: float) -> None:
                    self.root.after(0, lambda: self._update_progress(msg, pct * 100.0))

                res = self.pipeline.process_page(
                    image_path=self.current_image_path,
                    output_dir=OUTPUT_DIR,
                    save_debug_artifacts=True,
                    progress_callback=prog,
                )
                self.root.after(0, lambda: self._on_page_processed(res))
            except Exception as exc:
                self.root.after(0, lambda: messagebox.showerror("İşlem Hatası", str(exc)))
                self.root.after(0, lambda: self._update_progress(f"Hata: {exc}", 0.0))
            finally:
                self._is_busy = False

        threading.Thread(target=worker, daemon=True).start()

    def _update_progress(self, msg: str, pct: float) -> None:
        self.status_var.set(msg)
        self.progress_var.set(pct)

    def _on_page_processed(self, result: PagePipelineResult) -> None:
        self.current_result = result
        self._populate_block_tree()
        self.view_mode.set("final_tr")
        self._refresh_canvas_image()
        self.status_var.set(
            f"Tamamlandı: {len(result.blocks)} balon çevrildi → {result.output_image_path}"
        )

    def _populate_block_tree(self) -> None:
        self.block_tree.delete(*self.block_tree.get_children())
        if not self.current_result:
            return
        for idx, blk in enumerate(self.current_result.blocks):
            self.block_tree.insert(
                "",
                tk.END,
                iid=str(idx),
                values=(f"{blk['id']:02d}", blk.get("text", ""), blk.get("translation", "")),
            )
        if self.current_result.blocks:
            self.block_tree.selection_set("0")
            self._select_block_by_index(0)

    def on_tree_select(self, _event: tk.Event) -> None:
        sel = self.block_tree.selection()
        if not sel:
            return
        self._select_block_by_index(int(sel[0]))

    def _select_block_by_index(self, idx: int) -> None:
        if not self.current_result or not (0 <= idx < len(self.current_result.blocks)):
            return
        self.selected_block_idx = idx
        blk = self.current_result.blocks[idx]

        self.selected_label.configure(
            text=f"Seçili Balon #{blk['id']:02d}  (OCR Güven: {blk.get('ocr_confidence', 0):.2f} | Mod: {blk.get('clean_mode', '')})"
        )
        self.en_text_widget.delete("1.0", tk.END)
        self.en_text_widget.insert("1.0", str(blk.get("text", "")))

        self.tr_text_widget.delete("1.0", tk.END)
        self.tr_text_widget.insert("1.0", str(blk.get("translation", "")))

        self.block_font_var.set(str(blk.get("font_name", "")))
        self.block_size_var.set(int(blk.get("render_font_size", 0) or 0))
        self._refresh_canvas_image()

    def on_retranslate_selected_block(self) -> None:
        if self.current_result is None or self.selected_block_idx is None:
            return
        en_txt = self.en_text_widget.get("1.0", tk.END).strip()
        if not en_txt:
            return
        self.config.translator.backend = self.backend_var.get()
        tr_txt = self.pipeline.translator.translate_single(
            en_txt, custom_glossary=self.pipeline.project.glossary
        )
        self.tr_text_widget.delete("1.0", tk.END)
        self.tr_text_widget.insert("1.0", tr_txt)
        self.on_apply_block_edit()

    def on_apply_block_edit(self) -> None:
        if self.current_result is None or self.selected_block_idx is None:
            return
        idx = self.selected_block_idx
        blk = self.current_result.blocks[idx]

        blk["text"] = self.en_text_widget.get("1.0", tk.END).strip()
        blk["translation"] = self.tr_text_widget.get("1.0", tk.END).strip()
        blk["font_name"] = self.block_font_var.get().strip()
        blk["render_font_size"] = int(self.block_size_var.get() or 0)

        self.block_tree.item(
            str(idx),
            values=(f"{blk['id']:02d}", blk["text"], blk["translation"]),
        )

        self.pipeline.rerender_typesetting(self.current_result)
        if self.view_mode.get() not in ("final_tr", "side_by_side"):
            self.view_mode.set("final_tr")
        self._refresh_canvas_image()
        self.status_var.set(
            f"Balon #{blk['id']:02d} güncellendi ve yeniden basıldı ({self.current_result.output_image_path})"
        )

    def on_export_image(self) -> None:
        if self.current_result is None:
            messagebox.showinfo("Bilgi", "Önce sayfayı çevirin.")
            return
        save_path = filedialog.asksaveasfilename(
            title="Türkçe Manga Sayfasını Kaydet",
            initialdir=str(OUTPUT_DIR),
            initialfile=f"{self.current_result.input_path.stem}_tr.jpg",
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")],
        )
        if save_path:
            self.pipeline._imwrite_unicode(
                Path(save_path),
                self.current_result.final_tr_bgr,
                quality=self.config.jpeg_quality,
            )
            self.status_var.set(f"Dışa aktarıldı: {save_path}")

    # ------------------------------------------------------------------
    # Glossary Modal & Multi-Chapter Batch Modal
    # ------------------------------------------------------------------

    def on_open_glossary_dialog(self) -> None:
        win = tk.Toplevel(self.root)
        win.title("Proje Terimler Sözlüğü (Bölüm Hafızası)")
        win.geometry("540x420")
        win.configure(bg=self.BG_PANEL)
        win.transient(self.root)

        ttk.Label(
            win,
            text="Bölümler Boyunca Sabit Kalacak Özel Terimler & İsimler",
            style="Header.TLabel",
        ).pack(anchor=tk.W, padx=12, pady=(12, 6))

        tree = ttk.Treeview(win, columns=("en", "tr"), show="headings", height=10)
        tree.heading("en", text="İngilizce Terim")
        tree.heading("tr", text="Türkçe Karşılığı")
        tree.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)

        def refresh_terms() -> None:
            tree.delete(*tree.get_children())
            for k, v in self.pipeline.project.glossary.items():
                tree.insert("", tk.END, values=(k, v))

        refresh_terms()

        form = ttk.Frame(win, padding=12)
        form.pack(fill=tk.X)
        en_var = tk.StringVar()
        tr_var = tk.StringVar()
        ttk.Entry(form, textvariable=en_var, width=22).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Entry(form, textvariable=tr_var, width=22).pack(side=tk.LEFT, padx=(0, 6))

        def add_term() -> None:
            if en_var.get().strip() and tr_var.get().strip():
                self.pipeline.project.add_term(en_var.get(), tr_var.get())
                en_var.set("")
                tr_var.set("")
                refresh_terms()

        def del_selected() -> None:
            for item in tree.selection():
                vals = tree.item(item, "values")
                if vals:
                    self.pipeline.project.remove_term(str(vals[0]))
            refresh_terms()

        ttk.Button(form, text="➕ Ekle", style="Accent.TButton", command=add_term).pack(side=tk.LEFT, padx=4)
        ttk.Button(form, text="🗑️ Seçileni Sil", command=del_selected).pack(side=tk.LEFT, padx=4)

    def on_open_api_settings_dialog(self) -> None:
        from app.config import ENV_FILE_PATH

        win = tk.Toplevel(self.root)
        win.title("Çeviri Motoru & API Ayarları (Opsiyonel)")
        win.geometry("560x380")
        win.configure(bg=self.BG_PANEL)
        win.transient(self.root)

        frm = ttk.Frame(win, padding=14)
        frm.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            frm,
            text="Not: 'auto' veya 'free_web' modunda hiçbir API anahtarına ihtiyaç olmadan ücretsiz çalışır.",
            style="Muted.TLabel",
        ).pack(anchor=tk.W, pady=(0, 10))

        gem_var = tk.StringVar(value=self.config.translator.gemini_api_key)
        oai_key_var = tk.StringVar(value=getattr(self.config.translator, "openai_api_key", ""))
        oai_url_var = tk.StringVar(value=self.config.translator.openai_compat_url)
        ollama_model_var = tk.StringVar(value=self.config.translator.ollama_model)

        ttk.Label(frm, text="Gemini API Key (GEMINI_API_KEY - Opsiyonel):", style="Header.TLabel").pack(anchor=tk.W)
        ttk.Entry(frm, textvariable=gem_var, show="*").pack(fill=tk.X, pady=(2, 8))

        ttk.Label(frm, text="OpenAI / DeepSeek / OpenRouter API Key (OPENAI_API_KEY - Opsiyonel):", style="Header.TLabel").pack(anchor=tk.W)
        ttk.Entry(frm, textvariable=oai_key_var, show="*").pack(fill=tk.X, pady=(2, 8))

        ttk.Label(frm, text="OpenAI Uyumlu Endpoint URL (OPENAI_COMPAT_URL):", style="Header.TLabel").pack(anchor=tk.W)
        ttk.Entry(frm, textvariable=oai_url_var).pack(fill=tk.X, pady=(2, 8))

        ttk.Label(frm, text="Yerel Ollama Model Adı (OLLAMA_MODEL):", style="Header.TLabel").pack(anchor=tk.W)
        ttk.Entry(frm, textvariable=ollama_model_var).pack(fill=tk.X, pady=(2, 12))

        def save_api_settings() -> None:
            self.config.translator.gemini_api_key = gem_var.get().strip()
            self.config.translator.openai_api_key = oai_key_var.get().strip()
            self.config.translator.openai_compat_url = oai_url_var.get().strip()
            self.config.translator.ollama_model = ollama_model_var.get().strip()

            env_lines = [
                "# MangaTR-Studio Yerel API Yapılandırması (.gitignore ile gizlenir)",
                f"MANGATR_BACKEND={self.backend_var.get()}",
                f"GEMINI_API_KEY={self.config.translator.gemini_api_key}",
                f"OPENAI_API_KEY={self.config.translator.openai_api_key}",
                f"OPENAI_COMPAT_URL={self.config.translator.openai_compat_url}",
                f"OLLAMA_MODEL={self.config.translator.ollama_model}",
            ]
            ENV_FILE_PATH.write_text("\n".join(env_lines) + "\n", encoding="utf-8")
            self.status_var.set("API ayarları güncellendi ve yerel .env dosyasına kaydedildi.")
            win.destroy()

        ttk.Button(frm, text="💾 Kaydet (.env)", style="Accent.TButton", command=save_api_settings).pack(anchor=tk.E)

    def on_open_batch_dialog(self) -> None:
        win = tk.Toplevel(self.root)
        win.title("Toplu Bölüm İşleme (1 - 400+ Bölüm Batch Çeviri)")
        win.geometry("640x360")
        win.configure(bg=self.BG_PANEL)
        win.transient(self.root)

        in_var = tk.StringVar(value=str(INPUT_DIR))
        out_var = tk.StringVar(value=str(OUTPUT_DIR / "batch"))
        skip_var = tk.BooleanVar(value=True)
        debug_var = tk.BooleanVar(value=False)
        batch_status = tk.StringVar(value="Klasör seçip 'Toplu Çeviriyi Başlat' düğmesine basın.")
        batch_prog = tk.DoubleVar(value=0.0)

        frm = ttk.Frame(win, padding=14)
        frm.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frm, text="Giriş Klasörü (Tek Bölüm veya 400 Bölümlük Ana Klasör):", style="Header.TLabel").pack(anchor=tk.W)
        r1 = ttk.Frame(frm)
        r1.pack(fill=tk.X, pady=(4, 10))
        ttk.Entry(r1, textvariable=in_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        ttk.Button(
            r1,
            text="Seç...",
            command=lambda: in_var.set(filedialog.askdirectory(initialdir=in_var.get()) or in_var.get()),
        ).pack(side=tk.RIGHT)

        ttk.Label(frm, text="Çıkış Klasörü:", style="Header.TLabel").pack(anchor=tk.W)
        r2 = ttk.Frame(frm)
        r2.pack(fill=tk.X, pady=(4, 10))
        ttk.Entry(r2, textvariable=out_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        ttk.Button(
            r2,
            text="Seç...",
            command=lambda: out_var.set(filedialog.askdirectory(initialdir=out_var.get()) or out_var.get()),
        ).pack(side=tk.RIGHT)

        opts = ttk.Frame(frm)
        opts.pack(fill=tk.X, pady=4)
        ttk.Checkbutton(opts, text="Daha önce çevrilmiş sayfaları atla (Kaldığı yerden devam et)", variable=skip_var).pack(anchor=tk.W)
        ttk.Checkbutton(opts, text="Debug görsellerini de kaydet (_cleaned, _detected)", variable=debug_var).pack(anchor=tk.W)

        ttk.Label(frm, textvariable=batch_status, style="Muted.TLabel").pack(anchor=tk.W, pady=(12, 4))
        ttk.Progressbar(frm, variable=batch_prog, maximum=100.0).pack(fill=tk.X, pady=(0, 12))

        btn_bar = ttk.Frame(frm)
        btn_bar.pack(fill=tk.X)

        def start_batch() -> None:
            if self._is_busy:
                return
            self._is_busy = True

            def run() -> None:
                try:
                    def cb(idx: int, total: int, p: Path, st: str) -> None:
                        pct = (idx / float(max(1, total))) * 100.0
                        msg = f"[{idx}/{total}] {p.parent.name}/{p.name} — {st}"
                        self.root.after(0, lambda: (batch_status.set(msg), batch_prog.set(pct)))

                    summary = self.batch_processor.run_batch(
                        input_root=in_var.get(),
                        output_root=out_var.get(),
                        save_debug_artifacts=debug_var.get(),
                        skip_existing=skip_var.get(),
                        on_page_progress=cb,
                    )
                    done_msg = (
                        f"Tamamlandı! Toplam: {summary.total_pages} | "
                        f"Başarılı: {summary.succeeded_pages} | Hatalı: {summary.failed_pages} "
                        f"({summary.elapsed_seconds} sn)"
                    )
                    self.root.after(0, lambda: batch_status.set(done_msg))
                finally:
                    self._is_busy = False

            threading.Thread(target=run, daemon=True).start()

        ttk.Button(btn_bar, text="🚀 Toplu Çeviriyi Başlat", style="Success.TButton", command=start_batch).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btn_bar, text="⏹ Durdur", command=self.batch_processor.request_cancel).pack(side=tk.LEFT)
