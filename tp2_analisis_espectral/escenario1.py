"""
Practica 2 - Escenario 1: analisis por bloques.

Librerias usadas:
pip install numpy scipy matplotlib pyedflib
"""

import numpy as np
import matplotlib.pyplot as plt

from spectral_common import (
    BANDS,
    RECORDS,
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


def imprimir_frecuencias_dominantes(nombre, freqs, spectrum):
    print(f"\nFrecuencias dominantes - {nombre}")
    for f, amp in frecuencias_dominantes(freqs, spectrum):
        print(f"  {f:6.2f} Hz  valor medio={amp:.6g}")


def escenario1_fft_psd(segmentos, fs, canal=0):
    bloques = ["before", "crisis", "after"]
    colores = {"before": "steelblue", "crisis": "tomato", "after": "seagreen"}

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=False)
    fig.suptitle(
        f"Escenario 1 - FFT y PSD por bloques - Canal {canal}",
        fontsize=13,
        fontweight="bold",
    )

    for name in bloques:
        seg = segmentos[name]
        f_fft, fft_mag = calcular_fft(seg, fs)
        f_psd, psd = calcular_psd_welch(seg, fs)

        imprimir_frecuencias_dominantes(f"{name.upper()} FFT", f_fft, fft_mag)
        imprimir_frecuencias_dominantes(f"{name.upper()} PSD", f_psd, psd)

        axes[0].plot(
            f_fft[f_fft <= 64],
            fft_mag[canal, f_fft <= 64],
            color=colores[name],
            alpha=0.85,
            label=name,
        )
        axes[1].semilogy(
            f_psd[f_psd <= 64],
            psd[canal, f_psd <= 64],
            color=colores[name],
            alpha=0.85,
            label=name,
        )

    axes[0].set_title("FFT: magnitud espectral")
    axes[0].set_ylabel("Magnitud")
    axes[0].grid(linestyle=":", alpha=0.4)
    axes[0].legend()

    axes[1].set_title("PSD por Welch: potencia por Hz")
    axes[1].set_xlabel("Frecuencia (Hz)")
    axes[1].set_ylabel("PSD")
    axes[1].grid(linestyle=":", alpha=0.4)
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def escenario1_bandas(segmentos, fs):
    bloques = ["before", "crisis", "after"]
    band_names = list(BANDS.keys())
    x = np.arange(len(band_names))
    width = 0.25

    resumen = {}
    for name in bloques:
        freqs, psd = calcular_psd_welch(segmentos[name], fs)
        powers = potencia_por_banda(freqs, psd)
        resumen[name] = np.array([powers[b].mean() for b in band_names])

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, resumen["before"], width, label="Before", color="steelblue")
    ax.bar(x, resumen["crisis"], width, label="Crisis", color="tomato")
    ax.bar(x + width, resumen["after"], width, label="After", color="seagreen")
    ax.set_title("Escenario 1 - Potencia media por banda cerebral")
    ax.set_ylabel("Potencia media PSD")
    ax.set_xticks(x)
    ax.set_xticklabels([b.upper() for b in band_names])
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend()
    plt.tight_layout()
    plt.show()

    print("\nPotencia media por banda (promedio de canales)")
    print(f"{'Banda':<10} {'Before':>12} {'Crisis':>12} {'After':>12} {'Crisis/Before':>15}")
    for i, band in enumerate(band_names):
        before = resumen["before"][i]
        crisis = resumen["crisis"][i]
        after = resumen["after"][i]
        print(f"{band:<10} {before:>12.4g} {crisis:>12.4g} {after:>12.4g} {crisis / (before + 1e-12):>15.2f}x")


def resumen_espectral_por_bloque(freqs, spectrum):
    mask = (freqs > 0) & (freqs <= 64)
    freqs_lim = freqs[mask]
    spec_lim = spectrum[:, mask]

    peak_idx = np.argmax(spec_lim, axis=1)
    peak_freq = freqs_lim[peak_idx]
    peak_power = spec_lim[np.arange(spec_lim.shape[0]), peak_idx]
    total_power = np.trapezoid(spec_lim, freqs_lim, axis=1)

    return {
        "peak_freq_mean": float(np.mean(peak_freq)),
        "peak_power_mean": float(np.mean(peak_power)),
        "total_power_mean": float(np.mean(total_power)),
        "best_channel": int(np.argmax(total_power)),
        "best_channel_power": float(np.max(total_power)),
    }


def escenario1_periodograma(segmentos, fs, labels, canal=0):
    bloques = ["before", "crisis", "after"]
    colores = {"before": "steelblue", "crisis": "tomato", "after": "seagreen"}
    resumen = {}

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle(
        f"Escenario 1 - Periodograma por bloques - Canal {canal}",
        fontsize=13,
        fontweight="bold",
    )

    for name in bloques:
        freqs, pxx = calcular_periodograma(segmentos[name], fs)
        resumen[name] = resumen_espectral_por_bloque(freqs, pxx)
        ax.semilogy(
            freqs[freqs <= 64],
            pxx[canal, freqs <= 64],
            color=colores[name],
            alpha=0.85,
            label=name,
        )

    ax.set_title("Periodograma: estimacion directa de densidad espectral")
    ax.set_xlabel("Frecuencia (Hz)")
    ax.set_ylabel("Potencia/Hz")
    ax.grid(linestyle=":", alpha=0.4)
    ax.legend()
    plt.tight_layout()
    plt.show()

    print("\nEscenario 1 - Periodograma")
    print(f"{'Bloque':<10} {'Frec pico prom':>15} {'Pot total prom':>16} {'Canal max':>12} {'Pot canal max':>16}")
    for name in bloques:
        r = resumen[name]
        label = labels[r["best_channel"]] if r["best_channel"] < len(labels) else r["best_channel"]
        print(
            f"{name:<10} {r['peak_freq_mean']:>15.2f} {r['total_power_mean']:>16.4g} "
            f"{str(label):>12} {r['best_channel_power']:>16.4g}"
        )


def resumen_stft_por_bloque(freqs, power):
    band_power = potencia_bandas_tiempo(freqs, power)
    band_means = {band: float(values.mean()) for band, values in band_power.items()}
    best_band = max(band_means, key=band_means.get)
    values = band_power[best_band]
    channel_power = values.mean(axis=1)
    best_channel = int(np.argmax(channel_power))

    return {
        "band_means": band_means,
        "best_band": best_band,
        "best_channel": best_channel,
        "best_channel_power": float(channel_power[best_channel]),
    }


def escenario1_stft(segmentos, fs, labels, canal=0, window_sec=2, overlap=0.50):
    bloques = ["before", "crisis", "after"]
    resumen = {}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    fig.suptitle(
        f"Escenario 1 - STFT por bloques - Canal {canal} | ventana={window_sec}s overlap={overlap:.0%}",
        fontsize=13,
        fontweight="bold",
    )

    for ax, name in zip(axes, bloques):
        freqs, times, power = calcular_stft(segmentos[name], fs, window_sec, overlap)
        resumen[name] = resumen_stft_por_bloque(freqs, power)
        mask = freqs <= 64
        im = ax.pcolormesh(
            times,
            freqs[mask],
            10 * np.log10(power[canal, mask, :] + 1e-12),
            shading="auto",
            cmap="viridis",
        )
        ax.set_title(name.upper())
        ax.set_xlabel("Tiempo (s)")
        ax.grid(linestyle=":", alpha=0.25)

    axes[0].set_ylabel("Frecuencia (Hz)")
    fig.colorbar(im, ax=axes, label="Potencia (dB)")
    plt.show()

    print("\nEscenario 1 - STFT")
    print(f"{'Bloque':<10} {'Banda dominante':>16} {'Canal max':>12} {'Pot canal max':>16}")
    for name in bloques:
        r = resumen[name]
        label = labels[r["best_channel"]] if r["best_channel"] < len(labels) else r["best_channel"]
        print(f"{name:<10} {r['best_band']:>16} {str(label):>12} {r['best_channel_power']:>16.4g}")

    print("\nSTFT - Potencia media por banda")
    print(f"{'Banda':<10} {'Before':>12} {'Crisis':>12} {'After':>12} {'Crisis/Before':>15}")
    for band in BANDS:
        before = resumen["before"]["band_means"][band]
        crisis = resumen["crisis"]["band_means"][band]
        after = resumen["after"]["band_means"][band]
        print(f"{band:<10} {before:>12.4g} {crisis:>12.4g} {after:>12.4g} {crisis / (before + 1e-12):>15.2f}x")


def escenario1_scatter(segmentos, fs, labels):
    bloques = ["before", "crisis", "after"]
    colores = {"before": "steelblue", "crisis": "tomato", "after": "seagreen"}
    band_x = "beta"
    band_y = "gamma"
    resumen = {}

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.suptitle(
        f"Escenario 1 - Scatter por canal: potencia {band_x.upper()} vs {band_y.upper()}",
        fontsize=13,
        fontweight="bold",
    )

    for name in bloques:
        freqs, psd = calcular_psd_welch(segmentos[name], fs)
        powers = potencia_por_banda(freqs, psd)
        x = powers[band_x]
        y = powers[band_y]
        resumen[name] = {
            f"{band_x}_mean": float(np.mean(x)),
            f"{band_y}_mean": float(np.mean(y)),
            "ratio_gamma_beta": float(np.mean(y / (x + 1e-12))),
            "best_channel_gamma": int(np.argmax(y)),
        }
        ax.scatter(x, y, color=colores[name], alpha=0.75, s=45, label=name)

        for idx in np.argsort(y)[-2:]:
            ax.annotate(labels[idx], (x[idx], y[idx]), fontsize=8, alpha=0.8, xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel(f"Potencia {band_x.upper()} por canal")
    ax.set_ylabel(f"Potencia {band_y.upper()} por canal")
    ax.grid(linestyle=":", alpha=0.4)
    ax.legend()
    plt.tight_layout()
    plt.show()

    print("\nEscenario 1 - Scatter Beta vs Gamma")
    print(f"{'Bloque':<10} {'Beta prom':>12} {'Gamma prom':>12} {'Gamma/Beta':>12} {'Canal gamma max':>18}")
    for name in bloques:
        r = resumen[name]
        label = labels[r["best_channel_gamma"]] if r["best_channel_gamma"] < len(labels) else r["best_channel_gamma"]
        print(
            f"{name:<10} {r[f'{band_x}_mean']:>12.4g} {r[f'{band_y}_mean']:>12.4g} "
            f"{r['ratio_gamma_beta']:>12.2f} {str(label):>18}"
        )


def ejecutar_registro(record_key):
    cfg = RECORDS[record_key]
    if not find_edf_file(cfg["file"]).exists():
        print(f"\n[AVISO] Se saltea {record_key}: no se encontro {cfg['file']}")
        return

    print(f"\n{'=' * 80}")
    print(f"ESCENARIO 1 - {record_key}")
    print(f"{'=' * 80}")

    signals, fs, labels, _, segmentos, _ = cargar_datos(record_key)
    imprimir_resumen_carga(signals, fs, segmentos, record_key)
    escenario1_fft_psd(segmentos, fs, canal=0)
    escenario1_bandas(segmentos, fs)
    escenario1_periodograma(segmentos, fs, labels, canal=0)
    escenario1_stft(segmentos, fs, labels, canal=0, window_sec=2, overlap=0.50)
    escenario1_scatter(segmentos, fs, labels)


if __name__ == "__main__":
    for record_key in RECORDS:
        ejecutar_registro(record_key)
