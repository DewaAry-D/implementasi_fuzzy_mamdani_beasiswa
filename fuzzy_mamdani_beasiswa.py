"""
Fuzzy Mamdani - Sistem Kelayakan Penerima Beasiswa
Tahapan: fuzzifikasi, alpha-predikat (AND = MIN), implikasi MIN,
agregasi MAX, defuzzifikasi centroid diskrit.
"""
from itertools import product
import numpy as np


# ============================================================
# 1. FUNGSI KEANGGOTAAN (skalar)
# ============================================================

def left_shoulder(x, a, b):
    """Bahu kiri: 1 sampai a, turun linier, 0 mulai b."""
    if x <= a:
        return 1.0
    if x >= b:
        return 0.0
    return (b - x) / (b - a)


def right_shoulder(x, a, b):
    """Bahu kanan: 0 sampai a, naik linier, 1 mulai b."""
    if x <= a:
        return 0.0
    if x >= b:
        return 1.0
    return (x - a) / (b - a)


def triangle(x, a, b, c):
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (c - x) / (c - b)


def trapezoid(x, a, b, c, d):
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (d - x) / (d - c)


# ============================================================
# 2. DEFINISI HIMPUNAN FUZZY (satu sumber kebenaran)
# ============================================================
# Format: (bentuk, parameter)

MF_DEF = {
    "IPK": {
        "Rendah": ("left", (2.00, 2.75)),
        "Sedang": ("trap", (2.00, 2.60, 3.00, 3.50)),
        "Tinggi": ("right", (3.00, 3.50)),
    },
    "Penghasilan": {  # juta rupiah / bulan
        "Rendah": ("left", (2.5, 5.0)),
        "Sedang": ("trap", (2.5, 5.0, 8.0, 11.0)),
        "Tinggi": ("right", (8.0, 11.0)),
    },
    "Tanggungan": {
        "Sedikit": ("left", (2, 4)),
        "Sedang": ("tri", (1, 4, 7)),
        "Banyak": ("right", (4, 6)),
    },
    "Prestasi": {
        "Rendah": ("left", (30, 60)),
        "Sedang": ("tri", (30, 60, 90)),
        "Tinggi": ("right", (60, 80)),
    },
    "Kelayakan": {
        "Tidak Layak": ("left", (25, 45)),
        "Cukup Layak": ("tri", (30, 45, 60)),
        "Layak": ("tri", (50, 65, 80)),
        "Sangat Layak": ("right", (70, 85)),
    },
}

UNIVERSE = {
    "IPK": (0, 4),
    "Penghasilan": (0, 15),
    "Tanggungan": (0, 8),
    "Prestasi": (0, 100),
    "Kelayakan": (0, 100),
}

_SCALAR = {"left": left_shoulder, "right": right_shoulder,
           "tri": triangle, "trap": trapezoid}


def mf_scalar(variable, label, x):
    shape, params = MF_DEF[variable][label]
    return float(_SCALAR[shape](x, *params))


def mf_vector(variable, label, z):
    """Versi vektor (untuk grid output dan plot). Hasil identik dengan skalar."""
    shape, p = MF_DEF[variable][label]
    z = np.asarray(z, dtype=float)
    if shape == "left":
        return np.interp(z, [p[0], p[1]], [1.0, 0.0])
    if shape == "right":
        return np.interp(z, [p[0], p[1]], [0.0, 1.0])
    if shape == "tri":
        return np.interp(z, p, [0.0, 1.0, 0.0], left=0.0, right=0.0)
    return np.interp(z, p, [0.0, 1.0, 1.0, 0.0], left=0.0, right=0.0)


def fuzzify_variable(variable, x):
    return {label: mf_scalar(variable, label, x) for label in MF_DEF[variable]}


def membership_output(z):
    return fuzzify_variable("Kelayakan", z)


# ============================================================
# 3. BASIS ATURAN (20 inti + 61 penutup = 81)
# ============================================================

IPK_CATS = tuple(MF_DEF["IPK"])
PENGHASILAN_CATS = tuple(MF_DEF["Penghasilan"])
TANGGUNGAN_CATS = tuple(MF_DEF["Tanggungan"])
PRESTASI_CATS = tuple(MF_DEF["Prestasi"])

# (IPK, Penghasilan, Tanggungan, Prestasi): Konsekuen
CORE_RULES = {
    ("Tinggi", "Rendah", "Banyak", "Tinggi"): "Sangat Layak",  # R1
    ("Tinggi", "Rendah", "Sedang", "Tinggi"): "Sangat Layak",  # R2
    ("Tinggi", "Rendah", "Banyak", "Sedang"): "Layak",         # R3
    ("Tinggi", "Rendah", "Banyak", "Rendah"): "Layak",         # R4
    ("Tinggi", "Sedang", "Sedang", "Sedang"): "Layak",         # R5
    ("Tinggi", "Sedang", "Sedikit", "Tinggi"): "Layak",        # R6
    ("Tinggi", "Tinggi", "Sedikit", "Rendah"): "Cukup Layak",  # R7
    ("Tinggi", "Tinggi", "Sedikit", "Tinggi"): "Cukup Layak",  # R8
    ("Sedang", "Rendah", "Banyak", "Sedang"): "Layak",         # R9
    ("Sedang", "Rendah", "Sedang", "Sedang"): "Cukup Layak",   # R10
    ("Sedang", "Sedang", "Sedang", "Sedang"): "Cukup Layak",   # R11
    ("Sedang", "Tinggi", "Sedikit", "Sedang"): "Tidak Layak",  # R12
    ("Sedang", "Tinggi", "Sedikit", "Rendah"): "Tidak Layak",  # R13
    ("Rendah", "Rendah", "Banyak", "Tinggi"): "Cukup Layak",   # R14
    ("Rendah", "Rendah", "Banyak", "Rendah"): "Tidak Layak",   # R15
    ("Rendah", "Tinggi", "Sedikit", "Tinggi"): "Tidak Layak",  # R16
    ("Rendah", "Sedang", "Sedang", "Sedang"): "Tidak Layak",   # R17
    ("Rendah", "Tinggi", "Sedikit", "Rendah"): "Tidak Layak",  # R18
    ("Sedang", "Rendah", "Sedikit", "Tinggi"): "Cukup Layak",  # R19
    ("Sedang", "Rendah", "Banyak", "Tinggi"): "Layak",         # R20
}


def closing_rule_consequent(ipk, penghasilan, tanggungan, prestasi):
    """
    Aturan penutup berbasis skor: Rendah/Sedikit=0, Sedang=1, Tinggi/Banyak=2;
    penghasilan dibalik (Rendah=2, Sedang=1, Tinggi=0).
    Total 0-2: Tidak Layak; 3-4: Cukup Layak; 5-6: Layak; 7-8: Sangat Layak.
    """
    s = {"Rendah": 0, "Sedikit": 0, "Sedang": 1, "Tinggi": 2, "Banyak": 2}
    total = s[ipk] + (2 - s[penghasilan]) + s[tanggungan] + s[prestasi]
    if total <= 2:
        return "Tidak Layak"
    if total <= 4:
        return "Cukup Layak"
    if total <= 6:
        return "Layak"
    return "Sangat Layak"


def build_rule_base():
    rules = []
    for combo in product(IPK_CATS, PENGHASILAN_CATS,
                         TANGGUNGAN_CATS, PRESTASI_CATS):
        if combo in CORE_RULES:
            consequent, rule_type = CORE_RULES[combo], "inti"
        else:
            consequent, rule_type = closing_rule_consequent(*combo), "penutup"
        rules.append({"antecedent": combo, "consequent": consequent,
                      "type": rule_type})
    return rules


RULES = build_rule_base()


# ============================================================
# 4. FUZZIFIKASI
# ============================================================

def fuzzification(inputs):
    return {
        "IPK": fuzzify_variable("IPK", inputs["ipk"]),
        "Penghasilan": fuzzify_variable("Penghasilan", inputs["penghasilan"]),
        "Tanggungan": fuzzify_variable("Tanggungan", inputs["tanggungan"]),
        "Prestasi": fuzzify_variable("Prestasi", inputs["prestasi"]),
    }


# ============================================================
# 5. INFERENSI: AND MIN, IMPLIKASI MIN, AGREGASI MAX
# ============================================================

def mamdani_inference(fuzzy_inputs, output_grid):
    aggregated = np.zeros_like(output_grid, dtype=float)
    active_rules = []
    # Keanggotaan konsekuen dihitung sekali untuk seluruh grid.
    out_mf = {lab: mf_vector("Kelayakan", lab, output_grid)
              for lab in MF_DEF["Kelayakan"]}

    for number, rule in enumerate(RULES, start=1):
        ipk_c, inc_c, dep_c, ach_c = rule["antecedent"]
        alpha = min(
            fuzzy_inputs["IPK"][ipk_c],
            fuzzy_inputs["Penghasilan"][inc_c],
            fuzzy_inputs["Tanggungan"][dep_c],
            fuzzy_inputs["Prestasi"][ach_c],
        )
        if alpha <= 0:
            continue
        implied = np.minimum(alpha, out_mf[rule["consequent"]])
        aggregated = np.maximum(aggregated, implied)
        active_rules.append({
            "number": number,
            "antecedent": rule["antecedent"],
            "consequent": rule["consequent"],
            "alpha": alpha,
            "type": rule["type"],
            "implied": implied,
        })
    return aggregated, active_rules


# ============================================================
# 6. DEFUZZIFIKASI CENTROID DISKRIT
# ============================================================

def centroid_discrete(output_grid, aggregated):
    numerator = float(np.sum(output_grid * aggregated))
    denominator = float(np.sum(aggregated))
    if denominator == 0:
        raise ValueError(
            "Σμ_agregat = 0. Tidak ada aturan aktif; periksa fungsi "
            "keanggotaan dan kelengkapan basis aturan."
        )
    return numerator / denominator, numerator, denominator


def classify_output(score):
    memberships = membership_output(score)
    category = max(memberships, key=memberships.get)
    return category, memberships


# ============================================================
# 7. FUNGSI UTAMA
# ============================================================

LIMITS = {"ipk": (0, 4), "penghasilan": (0, 15),
          "tanggungan": (0, 8), "prestasi": (0, 100)}


def evaluate_student(inputs, show_detail=True):
    for key, (lo, hi) in LIMITS.items():
        if not lo <= inputs[key] <= hi:
            raise ValueError(f"{key} harus berada pada rentang {lo} sampai {hi}.")

    # Grid centroid: 0..100, Δz = 0.1 (1.001 titik)
    output_grid = np.round(np.arange(0, 100.0 + 0.1, 0.1), 1)

    fuzzy_inputs = fuzzification(inputs)
    aggregated, active_rules = mamdani_inference(fuzzy_inputs, output_grid)
    score, num, den = centroid_discrete(output_grid, aggregated)
    category, out_mem = classify_output(score)

    result = {
        "input": inputs,
        "fuzzification": fuzzy_inputs,
        "active_rules": active_rules,
        "output_grid": output_grid,
        "aggregated": aggregated,
        "centroid_numerator": num,
        "centroid_denominator": den,
        "score": score,
        "category": category,
        "output_memberships": out_mem,
    }
    if show_detail:
        print_report(result)
    return result


def print_report(result):
    print("\n" + "=" * 65)
    print("HASIL PERHITUNGAN FUZZY MAMDANI – KELAYAKAN BEASISWA")
    print("=" * 65)

    print("\nA. DATA INPUT CRISP")
    for k, v in result["input"].items():
        print(f"{k:15s}: {v}")

    print("\nB. FUZZIFIKASI")
    for var, mem in result["fuzzification"].items():
        print(f"{var}:")
        for cat, deg in mem.items():
            print(f"  μ({cat:12s}) = {deg:.6f}")

    print("\nC. α-PREDIKAT (AND = MIN), ATURAN AKTIF")
    if not result["active_rules"]:
        print("Tidak ada aturan aktif.")
    for r in result["active_rules"]:
        print(f"Aturan ke-{r['number']:02d} | {r['antecedent']} -> "
              f"{r['consequent']} | α = {r['alpha']:.6f} | {r['type']}")

    print("\nD. IMPLIKASI MIN DAN AGREGASI MAX")
    print(f"Jumlah aturan aktif: {len(result['active_rules'])}")

    print("\nE. DEFUZZIFIKASI CENTROID DISKRIT")
    print("Grid: z = 0.0 sampai 100.0, Δz = 0.1, jumlah titik = 1001")
    print(f"Σ(z_i × μ_agregat(z_i)) = {result['centroid_numerator']:.8f}")
    print(f"Σμ_agregat(z_i)          = {result['centroid_denominator']:.8f}")
    print(f"Skor = {result['score']:.6f}")

    print("\nF. KATEGORI OUTPUT")
    for cat, deg in result["output_memberships"].items():
        print(f"μ({cat:14s}) = {deg:.6f}")
    print(f"Rekomendasi kategori: {result['category']}")
    print("=" * 65)


if __name__ == "__main__":
    mahasiswa = {"ipk": 3.20, "penghasilan": 3.00,
                 "tanggungan": 5, "prestasi": 75}
    evaluate_student(mahasiswa, show_detail=True)
