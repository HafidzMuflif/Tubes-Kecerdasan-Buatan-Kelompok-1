from google.colab import files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

uploaded = files.upload()
filename = list(uploaded.keys())[0]

df = pd.read_csv(filename)
df.head()


def trapmf(x, a, b, c, d):
    x = np.array(x, dtype=float)
    mu = np.zeros_like(x)

    if a != b:
        idx = (a < x) & (x < b)
        mu[idx] = (x[idx] - a) / (b - a)

    idx = (b <= x) & (x <= c)
    mu[idx] = 1.0

    if c != d:
        idx = (c < x) & (x < d)
        mu[idx] = (d - x[idx]) / (d - c)

    return np.clip(mu, 0, 1)


def plot_membership_monthly_hours():
    x = np.linspace(0, 1000, 1000)

    sebentar = trapmf(x, 0, 0, 200, 400)
    sedang   = trapmf(x, 200, 400, 600, 800)
    lama     = trapmf(x, 600, 800, 1000, 1000)

    plt.figure(figsize=(8,4))
    plt.plot(x, sebentar, label='Sebentar')
    plt.plot(x, sedang, label='Sedang')
    plt.plot(x, lama, label='Lama')

    plt.xlabel('Monthly Hours')
    plt.ylabel('μ')
    plt.title('Fungsi Keanggotaan Monthly Hours')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_membership_tariff_rate():
    t = np.linspace(0, 10, 1000)

    rendah   = trapmf(t, 0, 0, 3, 5)
    menengah = trapmf(t, 3, 5, 6, 8)
    tinggi   = trapmf(t, 6, 8, 10, 10)

    plt.figure(figsize=(8,4))
    plt.plot(t, rendah, label='Rendah')
    plt.plot(t, menengah, label='Menengah')
    plt.plot(t, tinggi, label='Tinggi')

    plt.xlabel('Tariff Rate')
    plt.ylabel('μ')
    plt.title('Fungsi Keanggotaan Tariff Rate')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_membership_electricity_bill():
    y = np.linspace(0, 10000, 1000)

    hemat = trapmf(y, 0, 0, 3000, 5000)
    cukup = trapmf(y, 3000, 5000, 6000, 8000)
    boros = trapmf(y, 6000, 8000, 10000, 10000)

    plt.figure(figsize=(8,4))
    plt.plot(y, hemat, label='Hemat')
    plt.plot(y, cukup, label='Cukup')
    plt.plot(y, boros, label='Boros')

    plt.xlabel('Electricity Bill')
    plt.ylabel('μ')
    plt.title('Fungsi Keanggotaan Electricity Bill')
    plt.legend()
    plt.grid(True)
    plt.show()


def fuzzy_monthly_hours(x):
    return {
        'Sebentar': trapmf(x, 0, 0, 200, 400),
        'Sedang'  : trapmf(x, 200, 400, 600, 800),
        'Lama'    : trapmf(x, 600, 800, 1000, 1000)
    }

def fuzzy_tariff(t):
    return {
        'Rendah'  : trapmf(t, 0, 0, 3, 5),
        'Menengah': trapmf(t, 3, 5, 6, 8),
        'Tinggi'  : trapmf(t, 6, 8, 10, 10)
    }


def mamdani_inference(mh, tr):
    rules = {
        ('Sebentar','Rendah'): 'Hemat',
        ('Sebentar','Menengah'): 'Hemat',
        ('Sebentar','Tinggi'): 'Cukup',
        ('Sedang','Rendah'): 'Hemat',
        ('Sedang','Menengah'): 'Cukup',
        ('Sedang','Tinggi'): 'Boros',
        ('Lama','Rendah'): 'Cukup',
        ('Lama','Menengah'): 'Boros',
        ('Lama','Tinggi'): 'Boros'
    }

    mh_f = fuzzy_monthly_hours(mh)
    tr_f = fuzzy_tariff(tr)

    out = {'Hemat':0, 'Cukup':0, 'Boros':0}

    for (a,b), c in rules.items():
        strength = min(mh_f[a], tr_f[b])
        out[c] = max(out[c], strength)

    return out


import pandas as pd

def tampilkan_tabel_inferensi_sederhana():
    data = [
        ['Sebentar', 'Rendah',   'Hemat'],
        ['Sebentar', 'Menengah', 'Hemat'],
        ['Sebentar', 'Tinggi',   'Cukup'],
        ['Sedang',   'Rendah',   'Hemat'],
        ['Sedang',   'Menengah', 'Cukup'],
        ['Sedang',   'Tinggi',   'Boros'],
        ['Lama',     'Rendah',   'Cukup'],
        ['Lama',     'Menengah', 'Boros'],
        ['Lama',     'Tinggi',   'Boros']
    ]

    df_rules = pd.DataFrame(
        data,
        columns=[
            'Monthly Hours',
            'Tariff Rate',
            'Electricity Bill'
        ]
    )

    print("\n=== TABEL ATURAN INFERENSI FUZZY ===")
    display(df_rules)


def defuzz_mamdani(out):
    y = np.linspace(0, 10000, 1000)

    mu_h = np.minimum(out['Hemat'], trapmf(y, 0, 0, 3000, 5000))
    mu_c = np.minimum(out['Cukup'], trapmf(y, 3000, 5000, 6000, 8000))
    mu_b = np.minimum(out['Boros'], trapmf(y, 6000, 8000, 10000, 10000))

    agg = np.maximum(mu_h, np.maximum(mu_c, mu_b))
    return np.sum(y * agg) / np.sum(agg)

def sugeno(out):
    z = {'Hemat':4000, 'Cukup':6000, 'Boros':8000}
    return sum(out[k]*z[k] for k in out) / sum(out.values())


def keputusan_akhir(y_crisp):
    if y_crisp <= 5000:
        return "Hemat"
    elif y_crisp <= 8000:
        return "Cukup"
    else:
        return "Boros"


plot_membership_monthly_hours()


plot_membership_tariff_rate()


plot_membership_electricity_bill()


tampilkan_tabel_inferensi_sederhana()


monthly_hours = float(input("Masukkan Monthly Hours (0–1000): "))
tariff_rate  = float(input("Masukkan Tariff Rate (0–10): "))


out = mamdani_inference(monthly_hours, tariff_rate)

print("\n=== NILAI FUZZY OUTPUT ===")
for k,v in out.items():
    print(f"{k} : {v:.3f}")

y_mamdani = defuzz_mamdani(out)
y_sugeno  = sugeno(out)

print("\n=== NILAI CRISP ===")
print("Mamdani y* :", round(y_mamdani, 2))
print("Sugeno  y* :", round(y_sugeno, 2))

print("\n=== KEPUTUSAN AKHIR ===")
print("Kategori (Mamdani):", keputusan_akhir(y_mamdani))
print("Kategori (Sugeno) :", keputusan_akhir(y_sugeno))


def plot_mamdani(out):
    y = np.linspace(0, 10000, 1000)

    mu_h = np.minimum(out['Hemat'], trapmf(y, 0, 0, 3000, 5000))
    mu_c = np.minimum(out['Cukup'], trapmf(y, 3000, 5000, 6000, 8000))
    mu_b = np.minimum(out['Boros'], trapmf(y, 6000, 8000, 10000, 10000))

    agg = np.maximum(mu_h, np.maximum(mu_c, mu_b))

    sample_x = np.linspace(500, 9500, 10)
    sample_y = np.interp(sample_x, y, agg)

    plt.figure(figsize=(8,4))
    plt.plot(y, mu_h, label='Hemat')
    plt.plot(y, mu_c, label='Cukup')
    plt.plot(y, mu_b, label='Boros')
    plt.plot(y, agg, 'k', linewidth=2, label='Agregasi')
    plt.scatter(sample_x, sample_y, color='black')

    plt.xlabel('Electricity Bill')
    plt.ylabel('μ')
    plt.title('Grafik Mamdani (Input User)')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_sugeno(out):
    z = {'Hemat':4000, 'Cukup':6000, 'Boros':8000}

    plt.figure(figsize=(8,4))
    for label in out:
        plt.vlines(z[label], 0, out[label], linewidth=4, label=label)

    plt.xlabel('Electricity Bill')
    plt.ylabel('μ')
    plt.title('Grafik Sugeno (Input User)')
    plt.legend()
    plt.grid(True)
    plt.show()


plot_mamdani(out)


plot_sugeno(out)
