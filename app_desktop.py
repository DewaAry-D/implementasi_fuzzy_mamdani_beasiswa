"""
Antarmuka Desktop Native (Tkinter) - Fuzzy Mamdani Kelayakan Beasiswa.
Jalankan langsung dengan klik tombol Run di VS Code atau:
python app_desktop.py
"""
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import fuzzy_mamdani_beasiswa as fm

COLORS = ["#2a9d8f", "#e9c46a", "#e76f51", "#6a4c93"]
CAT_COLOR = {
    "Tidak Layak": "#d62828",
    "Cukup Layak": "#e76f51",
    "Layak": "#2a9d8f",
    "Sangat Layak": "#1d6fb8",
}


class BeasiswaFuzzyApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Sistem Kelayakan Penerima Beasiswa - Fuzzy Mamdani (Desktop)")
        self.geometry("1100x720")
        self.minsize(900, 600)

        # Style configuration
        self.style = ttk.Style(self)
        self.style.theme_use("clam")
        self.configure(bg="#f4f6f9")

        # State & Debounce Timer
        self._calc_timer = None
        self.last_res = None
        self.tab_dirty = {0: True, 1: True, 2: True, 3: True}

        # Layout Container
        main_paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # ----------------- PANEL KIRI: INPUT & RINGKASAN -----------------
        left_frame = ttk.Frame(main_paned, padding=12)
        main_paned.add(left_frame, weight=1)

        # Title
        ttk.Label(left_frame, text="🎓 Data Mahasiswa", font=("Arial", 14, "bold")).pack(anchor="w", pady=(0, 10))

        # Input Variables
        self.val_ipk = tk.DoubleVar(value=3.20)
        self.val_penghasilan = tk.DoubleVar(value=3.0)
        self.val_tanggungan = tk.IntVar(value=5)
        self.val_prestasi = tk.IntVar(value=75)

        self._create_slider(left_frame, "IPK (0.00 - 4.00):", self.val_ipk, 0.0, 4.0, 0.01, "{:.2f}")
        self._create_slider(left_frame, "Penghasilan (juta Rp/bulan):", self.val_penghasilan, 0.0, 15.0, 0.1, "{:.1f}")
        self._create_slider(left_frame, "Jumlah Tanggungan (orang):", self.val_tanggungan, 0, 8, 1, "{:d}")
        self._create_slider(left_frame, "Skor Prestasi (0 - 100):", self.val_prestasi, 0, 100, 1, "{:d}")

        btn_calc = tk.Button(
            left_frame, text="⚡ Hitung Kelayakan", font=("Arial", 11, "bold"),
            bg="#2a9d8f", fg="white", activebackground="#21867a", activeforeground="white",
            relief=tk.FLAT, pady=6, cursor="hand2", command=self.calculate
        )
        btn_calc.pack(fill=tk.X, pady=(15, 15))

        # Result Card
        self.card_frame = tk.Frame(left_frame, bg="#ffffff", bd=1, relief=tk.SOLID, padx=12, pady=12)
        self.card_frame.pack(fill=tk.X, pady=5)

        self.lbl_card_title = tk.Label(self.card_frame, text="Hasil Evaluasi:", font=("Arial", 10), bg="#ffffff")
        self.lbl_card_title.pack(anchor="w")

        self.lbl_category = tk.Label(self.card_frame, text="-", font=("Arial", 16, "bold"), bg="#ffffff", fg="#2a9d8f")
        self.lbl_category.pack(anchor="w", pady=(4, 6))

        self.lbl_score = tk.Label(self.card_frame, text="Skor: -", font=("Arial", 10), bg="#ffffff")
        self.lbl_score.pack(anchor="w")

        self.lbl_rules = tk.Label(self.card_frame, text="Aturan aktif: -", font=("Arial", 10), bg="#ffffff")
        self.lbl_rules.pack(anchor="w")

        # ----------------- PANEL KANAN: DETAIL & GRAFIK (TABS) -----------------
        right_frame = ttk.Frame(main_paned, padding=8)
        main_paned.add(right_frame, weight=3)

        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        # Tab 1: Fuzzifikasi
        self.tab1 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab1, text="1. Fuzzifikasi")

        # Tab 2: Aturan Aktif
        self.tab2 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab2, text="2. Aturan Aktif")

        # Tab 3: Agregasi & Defuzzifikasi
        self.tab3 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab3, text="3. Agregasi & Defuzzifikasi")

        # Tab 4: Output MF
        self.tab4 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab4, text="4. Himpunan Output")

        # Setup canvas containers for matplotlib
        self.fig1, self.axs1 = plt.subplots(2, 2, figsize=(7.5, 5))
        self.fig1.tight_layout(pad=2.2)
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=self.tab1)
        self.canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Setup Table for Tab 2
        columns = ("No", "IPK", "Penghasilan", "Tanggungan", "Prestasi", "Konsekuen", "Alpha", "Jenis")
        self.tree_rules = ttk.Treeview(self.tab2, columns=columns, show="headings", height=15)
        for col in columns:
            self.tree_rules.heading(col, text=col)
            width = 50 if col in ("No", "Alpha") else 95
            self.tree_rules.column(col, width=width, anchor="center")

        scroll_y = ttk.Scrollbar(self.tab2, orient=tk.VERTICAL, command=self.tree_rules.yview)
        self.tree_rules.configure(yscrollcommand=scroll_y.set)
        self.tree_rules.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0), pady=10)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y, pady=10, padx=(0, 10))

        # Setup canvas for Tab 3
        self.fig3, self.ax3 = plt.subplots(figsize=(7.5, 4.2))
        self.canvas3 = FigureCanvasTkAgg(self.fig3, master=self.tab3)
        self.canvas3.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.lbl_defuzz_info = ttk.Label(self.tab3, text="", font=("Arial", 10))
        self.lbl_defuzz_info.pack(pady=8)

        # Setup canvas for Tab 4
        self.fig4, self.ax4 = plt.subplots(figsize=(7.5, 4.2))
        self.canvas4 = FigureCanvasTkAgg(self.fig4, master=self.tab4)
        self.canvas4.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Initial calculation
        self.calculate()

    def _create_slider(self, parent, label_text, variable, from_, to, step, fmt):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=6)

        lbl_header = ttk.Label(frame, text=label_text, font=("Arial", 9, "bold"))
        lbl_header.pack(anchor="w")

        sub_frame = ttk.Frame(frame)
        sub_frame.pack(fill=tk.X, pady=(2, 0))

        lbl_val = ttk.Label(sub_frame, text=fmt.format(variable.get()), width=7, font=("Arial", 9))
        lbl_val.pack(side=tk.RIGHT, padx=(4, 0))

        def on_slide(val):
            val_num = float(val)
            if isinstance(variable, tk.IntVar):
                val_num = int(round(val_num))
            elif step == 0.01:
                val_num = round(val_num, 2)
            elif step == 0.1:
                val_num = round(val_num, 1)

            variable.set(val_num)
            lbl_val.config(text=fmt.format(val_num))
            # Debounce calculation to keep slider completely smooth
            self._schedule_calculate()

        slider = ttk.Scale(sub_frame, from_=from_, to=to, variable=variable, command=on_slide)
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Saat tombol mouse dilepas setelah digeser, hitung langsung tanpa tunda
        slider.bind("<ButtonRelease-1>", lambda event: self.calculate())

    def _schedule_calculate(self, delay=75):
        """Debounce agar saat slider digeser cepat, UI tetap sangat ringan & responsif."""
        if self._calc_timer is not None:
            self.after_cancel(self._calc_timer)
        self._calc_timer = self.after(delay, self.calculate)

    def calculate(self):
        """Hitung logika fuzzy dan render hanya tab yang sedang aktif."""
        if self._calc_timer is not None:
            self.after_cancel(self._calc_timer)
            self._calc_timer = None

        inputs = {
            "ipk": float(self.val_ipk.get()),
            "penghasilan": float(self.val_penghasilan.get()),
            "tanggungan": int(self.val_tanggungan.get()),
            "prestasi": int(self.val_prestasi.get()),
        }

        try:
            res = fm.evaluate_student(inputs, show_detail=False)
        except Exception as e:
            messagebox.showerror("Error Perhitungan", str(e))
            return

        self.last_res = res
        cat = res["category"]
        score = res["score"]
        color = CAT_COLOR.get(cat, "#333333")

        # Update Card Ringkasan Hasil (Sangat cepat dan instan)
        self.lbl_category.config(text=cat, fg=color)
        self.lbl_score.config(text=f"Skor Centroid: {score:.3f} / 100")
        self.lbl_rules.config(text=f"Aturan aktif: {len(res['active_rules'])} dari {len(fm.RULES)}")
        self.card_frame.config(highlightbackground=color, highlightthickness=2)

        # Tandai semua tab butuh pembaruan
        for k in self.tab_dirty:
            self.tab_dirty[k] = True

        # Render HANYA tab yang sedang aktif dilihat user (Hemat beban render ~75%)
        try:
            active_tab = self.notebook.index(self.notebook.select())
        except Exception:
            active_tab = 0

        self._render_tab(active_tab)
        self.tab_dirty[active_tab] = False

    def _on_tab_changed(self, event=None):
        """Dipanggil saat user mengklik tab lain."""
        if self.last_res is None:
            return
        try:
            active_tab = self.notebook.index(self.notebook.select())
        except Exception:
            return

        if self.tab_dirty.get(active_tab, False):
            self._render_tab(active_tab)
            self.tab_dirty[active_tab] = False

    def _render_tab(self, tab_idx):
        res = self.last_res
        if res is None:
            return

        cat = res["category"]
        score = res["score"]
        color = CAT_COLOR.get(cat, "#333333")

        if tab_idx == 0:
            # ----------------- Tab 1: Fuzzifikasi -----------------
            key_map = [
                ("IPK", float(self.val_ipk.get())),
                ("Penghasilan", float(self.val_penghasilan.get())),
                ("Tanggungan", int(self.val_tanggungan.get())),
                ("Prestasi", int(self.val_prestasi.get())),
            ]

            for i, (var, x) in enumerate(key_map):
                ax = self.axs1[i // 2, i % 2]
                ax.clear()
                lo, hi = fm.UNIVERSE[var]
                grid = np.linspace(lo, hi, 200)

                for j, lab in enumerate(fm.MF_DEF[var]):
                    ax.plot(grid, fm.mf_vector(var, lab, grid), lw=1.8, color=COLORS[j], label=lab)
                    deg = res["fuzzification"][var][lab]
                    if deg > 0:
                        ax.plot([x], [deg], "o", color=COLORS[j], markersize=5)

                ax.axvline(x, color="black", ls="--", lw=1)
                ax.set_title(f"{var} = {x}", fontsize=9, fontweight="bold")
                ax.set_ylim(-0.02, 1.08)
                ax.tick_params(labelsize=8)
                ax.legend(loc="upper right", fontsize=7)
                ax.grid(alpha=0.25)

            self.fig1.tight_layout(pad=1.8)
            self.canvas1.draw_idle()

        elif tab_idx == 1:
            # ----------------- Tab 2: Aturan Aktif -----------------
            for row in self.tree_rules.get_children():
                self.tree_rules.delete(row)

            sorted_rules = sorted(res["active_rules"], key=lambda r: r["alpha"], reverse=True)
            for r in sorted_rules:
                self.tree_rules.insert("", tk.END, values=(
                    r["number"],
                    r["antecedent"][0],
                    r["antecedent"][1],
                    r["antecedent"][2],
                    r["antecedent"][3],
                    r["consequent"],
                    f"{r['alpha']:.4f}",
                    r["type"],
                ))

        elif tab_idx == 2:
            # ----------------- Tab 3: Agregasi & Defuzzifikasi -----------------
            z, agg = res["output_grid"], res["aggregated"]
            self.ax3.clear()
            for r in res["active_rules"]:
                self.ax3.plot(z, r["implied"], color="gray", alpha=0.3, lw=1)

            self.ax3.fill_between(z, agg, color=color, alpha=0.35, label="Agregasi MAX")
            self.ax3.plot(z, agg, color=color, lw=2)
            self.ax3.axvline(score, color="black", ls="--", lw=2, label=f"Centroid z* = {score:.3f}")
            self.ax3.set_xlabel("Skor Kelayakan (z)", fontsize=9)
            self.ax3.set_ylabel("μ agregat", fontsize=9)
            self.ax3.set_ylim(0, 1.05)
            self.ax3.grid(alpha=0.3)
            self.ax3.legend(fontsize=8)
            self.ax3.set_title("Kurva Implikasi & Agregasi Mamdani", fontsize=10, fontweight="bold")
            self.canvas3.draw_idle()

            self.lbl_defuzz_info.config(
                text=f"Centroid z* = ∑(z * μ) / ∑μ = {res['centroid_numerator']:.4f} / {res['centroid_denominator']:.4f} = {score:.4f}"
            )

        elif tab_idx == 3:
            # ----------------- Tab 4: Himpunan Output -----------------
            z_out = np.linspace(0, 100, 250)
            self.ax4.clear()
            for lab in fm.MF_DEF["Kelayakan"]:
                self.ax4.plot(z_out, fm.mf_vector("Kelayakan", lab, z_out), lw=2,
                              color=CAT_COLOR[lab], label=lab)

            self.ax4.axvline(score, color="black", ls="--", lw=1.5, label=f"Skor crisp ({score:.2f})")
            self.ax4.set_xlabel("Skor Kelayakan", fontsize=9)
            self.ax4.set_ylabel("μ", fontsize=9)
            self.ax4.grid(alpha=0.3)
            self.ax4.legend(ncol=4, loc="upper center", fontsize=8)
            self.ax4.set_ylim(0, 1.15)
            self.ax4.set_title("Fungsi Keanggotaan Kelayakan (Output)", fontsize=10, fontweight="bold")
            self.canvas4.draw_idle()


if __name__ == "__main__":
    app = BeasiswaFuzzyApp()
    app.mainloop()
