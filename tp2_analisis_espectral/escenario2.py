"""
Practica 2 - Escenario 2: analisis del bloque total [Before | Crisis | After].

Librerias usadas:
pip install numpy scipy matplotlib pyedflib
"""

import numpy as np
import matplotlib.pyplot as plt

from spectral_common import (
    BANDS,
    RECORDS,
    calcular_espectrograma,
    calcular_fft,
    calcular_periodograma,
    calcular_psd_welch,
    calcular_stft,
    cargar_datos,
    find_edf_file,
    frecuencias_dominantes,
    imprimir_resumen_carga,
    potencia_bandas_tiempo,
    potencia_por_banda,
)


def evaluar_config_espectrograma(total, fs, refs, window_sec, overlap):
    freqs, times, spec = calcular_espectrograma(total, fs, window_sec, overlap)
    powers = potencia_bandas_tiempo(freqs, spec)

    before_mask = times < refs["crisis_ini_total_sec"]
    crisis_mask = (times >= refs["crisis_ini_total_sec"]) & (times <= refs["crisis_fin_total_sec"])

    resultados = []
    for band, values in powers.items():
        before_mean = values[:, before_mask].mean(axis=1)
        crisis_mean = values[:, crisis_mask].mean(axis=1)
        ratio = crisis_mean / (before_mean + 1e-12)
        best_channel = int(np.argmax(ratio))
        resultados.append(
            {
                "window_sec": window_sec,
                "overlap": overlap,
                "band": band,
                "score": float(np.mean(ratio)),
                "best_channel": best_channel,
                "best_channel_score": float(ratio[best_channel]),
            }
        )
    return freqs, times, powers, resultados


def escenario2_espectrograma_bandas(segmentos, fs, refs, labels):
    """
    Prueba 3 tamanios de ventana y 3 overlaps.
    La mejor configuracion se elige por mayor aumento medio Crisis/Before.
    El panel superior es un heatmap (canales x tiempo), el inferior muestra la evolucion temporal.
    """
    window_options = [1, 2, 4]
    overlap_options = [0.25, 0.50, 0.75]
    total = segmentos["total"]

    all_results = []
    cache = {}
    for window_sec in window_options:
        for overlap in overlap_options:
            freqs, times, powers, results = evaluar_config_espectrograma(total, fs, refs, window_sec, overlap)
            key = (window_sec, overlap)
            cache[key] = (freqs, times, powers)
            all_results.extend(results)

    all_results.sort(key=lambda x: x["score"], reverse=True)
    best = all_results[0]
    best_key = (best["window_sec"], best["overlap"])
    _, times, powers = cache[best_key]

    print("\nEscenario 2 - Espectrograma por bandas: mejores configuraciones")
    print(f"{'Ventana':>8} {'Overlap':>8} {'Banda':>8} {'Score':>12} {'Canal':>10} {'Score canal':>14}")
    for row in all_results[:10]:
        label = labels[row["best_channel"]] if row["best_channel"] < len(labels) else row["best_channel"]
        print(
            f"{row['window_sec']:>8.1f} {row['overlap']:>8.0%} {row['band']:>8} "
            f"{row['score']:>12.2f} {str(label):>10} {row['best_channel_score']:>14.2f}"
        )

    band = best["band"]
    ch = best["best_channel"]
    values = powers[band]
    mean_band = values.mean(axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    fig.suptitle(
        "Escenario 2 - Espectrograma por bandas cerebrales\n"
        f"Mejor: {band.upper()} | ventana={best['window_sec']}s | overlap={best['overlap']:.0%} | canal={labels[ch]}",
        fontsize=13,
        fontweight="bold",
    )

    im = axes[0].imshow(
        10 * np.log10(values + 1e-12),
        aspect="auto",
        origin="lower",
        extent=[times[0], times[-1], 0, values.shape[0] - 1],
        cmap="viridis",
    )
    axes[0].axvspan(refs["crisis_ini_total_sec"], refs["crisis_fin_total_sec"],
                    color="red", alpha=0.18, label="Crisis")
    axes[0].set_ylabel("Canal")
    axes[0].set_title(f"Potencia {band.upper()} por canal y tiempo (dB) — Todos los canales")
    axes[0].legend(loc="upper right")
    fig.colorbar(im, ax=axes[0], label="dB")

    axes[1].plot(times, mean_band, color="steelblue", label="Promedio todos los canales")
    axes[1].plot(times, values[ch], color="tomato", label=f"Mejor canal: {labels[ch]}")
    axes[1].axvspan(refs["crisis_ini_total_sec"], refs["crisis_fin_total_sec"],
                    color="red", alpha=0.18, label="Crisis")
    axes[1].set_xlabel("Tiempo dentro del bloque total (s)")
    axes[1].set_ylabel(f"Potencia {band.upper()}")
    axes[1].set_title("Evolucion temporal de la banda mas discriminante")
    axes[1].grid(linestyle=":", alpha=0.4)
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    return best


def imprimir_bandas():
    print("\nBandas cerebrales usadas")
    for name, (fmin, fmax) in BANDS.items():
        print(f"  {name:<6}: {fmin:>2}-{fmax:<2} Hz")


def etiqueta_tiempo(t, refs):
    if t < refs["crisis_ini_total_sec"]:
        return "before"
    if t <= refs["crisis_fin_total_sec"]:
        return "crisis"
    return "after"


def crear_ventanas(total, fs, window_sec=1, step_sec=1):
    win_len = int(window_sec * fs)
    step = int(step_sec * fs)
    ventanas = []
    centros = []

    for start in range(0, total.shape[1] - win_len + 1, step):
        end = start + win_len
        ventanas.append(total[:, start:end])
        centros.append((start + win_len / 2) / fs)

    return np.array(ventanas), np.array(centros)


def imprimir_frecuencias_dominantes(nombre, freqs, spectrum):
    print(f"\nFrecuencias dominantes - {nombre}")
    for f, amp in frecuencias_dominantes(freqs, spectrum):
        print(f"  {f:6.2f} Hz  valor medio={amp:.6g}")


def escenario2_fft_psd_total(segmentos, fs, refs):
    """
    FFT y PSD del bloque total concatenado [Before | Crisis | After].
    Representacion: linea gruesa = media entre los 28 canales,
    banda sombreada = +/- 1 desv. estandar.
    """
    total = segmentos["total"]
    f_fft, fft_mag = calcular_fft(total, fs)
    f_psd, psd = calcular_psd_welch(total, fs)

    imprimir_frecuencias_dominantes("TOTAL FFT", f_fft, fft_mag)
    imprimir_frecuencias_dominantes("TOTAL PSD", f_psd, psd)

    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    fig.suptitle(
        "Escenario 2 - FFT y PSD del bloque total\n"
        "(linea gruesa: media entre canales | banda: ±1 desv. estandar)",
        fontsize=13,
        fontweight="bold",
    )

    mask_fft = f_fft <= 64
    mu_fft = fft_mag[:, mask_fft].mean(axis=0)
    sd_fft = fft_mag[:, mask_fft].std(axis=0)
    axes[0].fill_between(f_fft[mask_fft], mu_fft - sd_fft, mu_fft + sd_fft,
                         color="steelblue", alpha=0.30)
    axes[0].plot(f_fft[mask_fft], mu_fft, color="steelblue", linewidth=2.0, label="Media canales")
    axes[0].set_title("FFT del bloque total (media ± 1σ entre 28 canales)")
    axes[0].set_ylabel("Magnitud")
    axes[0].grid(linestyle=":", alpha=0.4)
    axes[0].legend()

    mask_psd = f_psd <= 64
    mu_psd = psd[:, mask_psd].mean(axis=0)
    sd_psd = psd[:, mask_psd].std(axis=0)
    axes[1].fill_between(f_psd[mask_psd],
                         np.maximum(mu_psd - sd_psd, 1e-12),
                         mu_psd + sd_psd,
                         color="tomato", alpha=0.30)
    axes[1].semilogy(f_psd[mask_psd], mu_psd, color="tomato", linewidth=2.0, label="Media canales")
    axes[1].set_title("PSD por Welch del bloque total (media ± 1σ)")
    axes[1].set_xlabel("Frecuencia (Hz)")
    axes[1].set_ylabel("PSD (V²/Hz)")
    axes[1].grid(linestyle=":", alpha=0.4)
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    print("\nEscenario 2 - FFT/PSD del bloque total")
    print(f"Duracion total analizada: {total.shape[1] / fs:.1f} s")
    print(f"Crisis dentro del bloque total: {refs['crisis_ini_total_sec']:.1f}s a {refs['crisis_fin_total_sec']:.1f}s")


def escenario2_ventanas_bandas(segmentos, fs, refs, labels, window_sec=1, step_sec=1):
    ventanas, centros = crear_ventanas(segmentos["total"], fs, window_sec, step_sec)
    band_names = list(BANDS.keys())
    band_matrix = {band: [] for band in band_names}
    gamma_channels = []

    for win in ventanas:
        freqs, psd = calcular_psd_welch(win, fs, nperseg=min(win.shape[1], fs), overlap=0)
        powers = potencia_por_banda(freqs, psd)
        for band in band_names:
            band_matrix[band].append(float(np.mean(powers[band])))
        gamma_channels.append(int(np.argmax(powers["gamma"])))

    for band in band_names:
        band_matrix[band] = np.array(band_matrix[band])
    gamma_channels = np.array(gamma_channels)

    labels_t = np.array([etiqueta_tiempo(t, refs) for t in centros])
    band_colors = {
        "delta": "#4c78a8",
        "theta": "#f58518",
        "alpha": "#54a24b",
        "beta": "#e45756",
        "gamma": "#8e6bbf",
    }

    fig, axes = plt.subplots(5, 1, figsize=(15, 12), sharex=True)
    fig.suptitle(
        f"Escenario 2 - Potencia media por banda en ventanas de {window_sec}s\n"
        "(promedio de todos los canales por ventana | zona roja: crisis)",
        fontsize=13,
        fontweight="bold",
    )

    for ax, band in zip(axes, band_names):
        ax.plot(centros, band_matrix[band], color=band_colors[band], linewidth=1.7)
        ax.fill_between(centros, band_matrix[band], color=band_colors[band], alpha=0.18)
        ax.axvspan(
            refs["crisis_ini_total_sec"],
            refs["crisis_fin_total_sec"],
            color="#ff6b6b",
            alpha=0.18,
            label="Crisis" if band == "delta" else None,
        )
        ax.axvline(refs["crisis_ini_total_sec"], color="#c92a2a", linestyle="--", linewidth=1)
        ax.axvline(refs["crisis_fin_total_sec"], color="#c92a2a", linestyle="--", linewidth=1)
        ax.set_ylabel(band.upper())
        ax.grid(linestyle=":", alpha=0.35)

    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("Tiempo dentro del bloque total (s)")
    plt.tight_layout()
    plt.show()

    print(f"\nEscenario 2 - Bandas por ventanas de {window_sec}s")
    print(f"{'Banda':<10} {'Before':>12} {'Crisis':>12} {'After':>12} {'Crisis/Before':>15}")
    for band in band_names:
        before = band_matrix[band][labels_t == "before"].mean()
        crisis = band_matrix[band][labels_t == "crisis"].mean()
        after = band_matrix[band][labels_t == "after"].mean()
        print(f"{band:<10} {before:>12.4g} {crisis:>12.4g} {after:>12.4g} {crisis / (before + 1e-12):>15.2f}x")

    unique, counts = np.unique(gamma_channels[labels_t == "crisis"], return_counts=True)
    order = np.argsort(counts)[::-1]
    print("\nCanales mas frecuentes con maxima potencia GAMMA durante crisis")
    for idx in order[:5]:
        ch = int(unique[idx])
        label = labels[ch] if ch < len(labels) else ch
        print(f"  {label:<10} {counts[idx]:>4} ventanas")

    return centros, labels_t, band_matrix


def escenario2_scatter_ventanas(centros, labels_t, band_matrix):
    colores = {"before": "#4c78a8", "crisis": "#e45756", "after": "#54a24b"}
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.suptitle(
        "Escenario 2 - Scatter por ventanas: potencia BETA vs GAMMA\n"
        "(cada punto = una ventana de 1s del bloque total)",
        fontsize=13,
        fontweight="bold",
    )

    for estado in ["before", "crisis", "after"]:
        mask = labels_t == estado
        ax.scatter(
            band_matrix["beta"][mask],
            band_matrix["gamma"][mask],
            color=colores[estado],
            edgecolor="white",
            linewidth=0.5,
            alpha=0.82,
            s=48,
            label=f"{estado} ({mask.sum()} ventanas)",
        )

    ax.set_xlabel("Potencia BETA promedio por ventana")
    ax.set_ylabel("Potencia GAMMA promedio por ventana")
    ax.grid(linestyle=":", alpha=0.4)
    ax.legend()
    plt.tight_layout()
    plt.show()

    print("\nEscenario 2 - Scatter por ventanas Beta vs Gamma")
    print(f"{'Estado':<10} {'Ventanas':>10} {'Beta prom':>12} {'Gamma prom':>12}")
    for estado in ["before", "crisis", "after"]:
        mask = labels_t == estado
        print(
            f"{estado:<10} {mask.sum():>10} {band_matrix['beta'][mask].mean():>12.4g} "
            f"{band_matrix['gamma'][mask].mean():>12.4g}"
        )


def escenario2_periodograma_stft(segmentos, fs, refs):
    """
    Periodograma del bloque total: linea = media, banda = +/- 1 sigma entre canales.
    STFT del bloque total: espectrograma usando el promedio de todos los canales.
    """
    total = segmentos["total"]
    freqs_p, pxx = calcular_periodograma(total, fs)
    freqs_s, times_s, power = calcular_stft(total, fs, window_sec=2, overlap=0.50)

    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    fig.suptitle(
        "Escenario 2 - Periodograma y STFT del bloque total\n"
        "(Periodograma: media ± 1σ entre canales | STFT: promedio de canales)",
        fontsize=13,
        fontweight="bold",
    )

    # Periodograma: media ± 1 sigma
    mask_p = freqs_p <= 64
    mu_pxx = pxx[:, mask_p].mean(axis=0)
    sd_pxx = pxx[:, mask_p].std(axis=0)
    axes[0].fill_between(freqs_p[mask_p],
                         np.maximum(mu_pxx - sd_pxx, 1e-12),
                         mu_pxx + sd_pxx,
                         color="steelblue", alpha=0.28)
    axes[0].semilogy(freqs_p[mask_p], mu_pxx, color="steelblue", linewidth=2.0, label="Media canales")
    axes[0].set_title("Periodograma del bloque total (media ± 1σ entre 28 canales)")
    axes[0].set_xlabel("Frecuencia (Hz)")
    axes[0].set_ylabel("Potencia/Hz")
    axes[0].grid(linestyle=":", alpha=0.4)
    axes[0].legend()

    # STFT: promedio de todos los canales
    mask_s = freqs_s <= 64
    mean_power = power[:, mask_s, :].mean(axis=0)
    im = axes[1].pcolormesh(
        times_s,
        freqs_s[mask_s],
        10 * np.log10(mean_power + 1e-12),
        shading="auto",
        cmap="viridis",
    )
    axes[1].axvspan(refs["crisis_ini_total_sec"], refs["crisis_fin_total_sec"],
                    color="red", alpha=0.18, label="Crisis")
    axes[1].set_title("STFT del bloque total (promedio de todos los canales)")
    axes[1].set_xlabel("Tiempo dentro del bloque total (s)")
    axes[1].set_ylabel("Frecuencia (Hz)")
    axes[1].legend()
    fig.colorbar(im, ax=axes[1], label="Potencia media (dB)")
    plt.tight_layout()
    plt.show()

    band_power = potencia_bandas_tiempo(freqs_s, power)
    labels_t = np.array([etiqueta_tiempo(t, refs) for t in times_s])

    print("\nEscenario 2 - STFT total: potencia media por banda")
    print(f"{'Banda':<10} {'Before':>12} {'Crisis':>12} {'After':>12} {'Crisis/Before':>15}")
    for band in BANDS:
        values = band_power[band].mean(axis=0)
        before = values[labels_t == "before"].mean()
        crisis = values[labels_t == "crisis"].mean()
        after = values[labels_t == "after"].mean()
        print(f"{band:<10} {before:>12.4g} {crisis:>12.4g} {after:>12.4g} {crisis / (before + 1e-12):>15.2f}x")


def ejecutar_registro(record_key):
    cfg = RECORDS[record_key]
    if not find_edf_file(cfg["file"]).exists():
        print(f"\n[AVISO] Se saltea {record_key}: no se encontro {cfg['file']}")
        return

    print(f"\n{'=' * 80}")
    print(f"ESCENARIO 2 - {record_key}")
    print(f"{'=' * 80}")

    signals, fs, labels, _, segmentos, refs = cargar_datos(record_key)
    imprimir_resumen_carga(signals, fs, segmentos, record_key)
    imprimir_bandas()
    escenario2_fft_psd_total(segmentos, fs, refs)
    centros, labels_t, band_matrix = escenario2_ventanas_bandas(segmentos, fs, refs, labels, window_sec=1, step_sec=1)
    escenario2_scatter_ventanas(centros, labels_t, band_matrix)
    escenario2_periodograma_stft(segmentos, fs, refs)
    escenario2_espectrograma_bandas(segmentos, fs, refs, labels)


if __name__ == "__main__":
    for record_key in RECORDS:
        ejecutar_registro(record_key)
