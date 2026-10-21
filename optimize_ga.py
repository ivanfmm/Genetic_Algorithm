import os
import json
import csv
import time
import random
import multiprocessing as mp
from pathlib import Path

import cv2
import numpy as np
from deap import base, creator, tools

from VO import DEFAULT_VO_PARAMS
from evaluator import KITTIDatasetCache, evaluate_vo_params


# ==================================================================
#  1. Espacio de Busqueda de los 12 Genes (Continuo en [0.0, 1.0])
# ==================================================================
# Formato: (nombre_param, min_val, max_val, es_entero)
GENE_SPECS = [
    ("orb_nfeatures",     500.0,  3000.0, True),
    ("orb_scaleFactor",   1.10,   1.50,   False),
    ("orb_nlevels",       4.0,    10.0,   True),
    ("orb_patchSize",     15.0,   35.0,   True),   # Se iguala tambien a orb_edgeThreshold
    ("orb_fastThreshold", 5.0,    25.0,   True),
    ("lowe_ratio",        0.60,   0.90,   False),
    ("disp_min",          1.0,    10.0,   False),
    ("pnp_iterations",    100.0,  500.0,  True),
    ("pnp_reproj_err",    0.50,   4.00,   False),
    ("min_pnp_inliers",   8.0,    40.0,   True),   # Se iguala tambien a min_pnp_matches
    ("floor_band",        0.20,   0.45,   False),
    ("floor_ransac_mm",   15.0,   80.0,   False),
]


def decode_individual(individual):
    """
    Convierte un cromosoma continuo en [0.0, 1.0]^12 al diccionario `params`
    con los tipos exactos (int/float) que espera VisualOdometry.
    """
    params = DEFAULT_VO_PARAMS.copy()
    for gene_val, (name, low, high, is_int) in zip(individual, GENE_SPECS):
        # Asegurar limites [0.0, 1.0]
        g = max(0.0, min(1.0, float(gene_val)))
        val = low + g * (high - low)
        if is_int:
            params[name] = int(round(val))
        else:
            params[name] = round(float(val), 4)

    # Reglas de acoplamiento para consistencia en OpenCV y PnP
    params["orb_edgeThreshold"] = params["orb_patchSize"]
    params["min_pnp_matches"]   = params["min_pnp_inliers"]
    return params


def encode_default_params():
    """
    Codifica DEFAULT_VO_PARAMS en [0.0, 1.0]^12 para inyectar la linea base
    en la poblacion inicial (Warm Start).
    """
    genes = []
    for name, low, high, _ in GENE_SPECS:
        val = float(DEFAULT_VO_PARAMS[name])
        g = (val - low) / (high - low)
        genes.append(max(0.0, min(1.0, g)))
    return genes


# ==================================================================
#  2. Manejo de Memoria para Multiprocessing en Windows
# ==================================================================
_WORKER_CACHE = None


def _init_worker(seq_dir, disp_dir, gt_poses_file, start_frame, num_frames):
    """
    Se ejecuta UNA sola vez por cada proceso hijo al crear el Pool.
    Carga el cache en la RAM local del worker y limita OpenCV a 1 hilo
    para evitar colisiones entre procesos.
    """
    global _WORKER_CACHE
    cv2.setNumThreads(1)
    _WORKER_CACHE = KITTIDatasetCache(
        seq_dir=seq_dir,
        disp_dir=disp_dir,
        gt_poses_file=gt_poses_file,
        start_frame=start_frame,
        num_frames=num_frames,
    )


def _evaluate_worker(individual):
    """Funcion que ejecuta cada worker para evaluar un individuo."""
    global _WORKER_CACHE
    params = decode_individual(individual)
    fitness, info = evaluate_vo_params(params, _WORKER_CACHE)
    return fitness, info


# ==================================================================
#  3. Configuracion de DEAP (Algoritmo Genetico Continuo)
# ==================================================================
# Evitar warning si se reimporta el modulo
if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin, metrics=dict)


def build_toolbox():
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, 0.0, 1.0)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_float, n=len(GENE_SPECS))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Operadores geneticos continuos acotados en [0.0, 1.0]
    # 1. Cruce Binario Simulado Acotado (SBX)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded,
                     low=0.0, up=1.0, eta=15.0)
    # 2. Mutacion Polinomial Acotada
    toolbox.register("mutate", tools.mutPolynomialBounded,
                     low=0.0, up=1.0, eta=20.0, indpb=0.25)
    # 3. Seleccion por Torneo
    toolbox.register("select", tools.selTournament, tournsize=3)
    return toolbox


# ==================================================================
#  4. Ciclo Evolutivo Principal con Elitismo y Checkpoints
# ==================================================================
def run_genetic_algorithm(
    seq_dir,
    disp_dir,
    gt_poses_file,
    start_frame=0,
    num_frames=1100,
    pop_size=20,
    n_generations=15,
    cx_prob=0.75,
    mut_prob=0.35,
    elite_size=2,
    n_workers=4,
    seed=42
):
    random.seed(seed)
    np.random.seed(seed)

    out_dir = Path(__file__).parent
    best_json_path = out_dir / "best_vo_params.json"
    csv_log_path   = out_dir / "ga_history.csv"

    toolbox = build_toolbox()

    # Crear poblacion inicial e inyectar el individuo Linea Base en la posicion 0
    pop = toolbox.population(n=pop_size)
    baseline_genes = encode_default_params()
    for i, g in enumerate(baseline_genes):
        pop[0][i] = g

    hof = tools.HallOfFame(maxsize=5)

    print("=" * 70)
    print(f" Iniciando Algoritmo Genetico Continuo (DEAP) - {num_frames} frames")
    print(f" Poblacion: {pop_size} | Generaciones: {n_generations} | Workers: {n_workers}")
    print("=" * 70)

    # Preparar archivo CSV de historial
    with open(csv_log_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "gen", "best_fitness", "best_ate_m", "best_rpe_trans_m",
            "best_rpe_rot_deg", "best_coverage", "mean_fitness", "gen_time_s"
        ])

    # Iniciar Pool de procesos con carga unica de datos por worker
    with mp.Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(seq_dir, disp_dir, gt_poses_file, start_frame, num_frames)
    ) as pool:

        for gen in range(n_generations + 1):
            t_gen = time.perf_counter()

            # Evaluar solo los individuos que no tienen fitness calculado
            invalid_ind = [ind for ind in pop if not ind.fitness.valid]
            if invalid_ind:
                results = pool.map(_evaluate_worker, invalid_ind)
                for ind, (fit, info) in zip(invalid_ind, results):
                    ind.fitness.values = (fit,)
                    ind.metrics = info

            # Actualizar Hall of Fame
            hof.update(pop)
            best_ind = hof[0]
            best_fit = best_ind.fitness.values[0]
            best_info = best_ind.metrics
            best_params = decode_individual(best_ind)

            valid_fits = [ind.fitness.values[0] for ind in pop if ind.fitness.values[0] < 1e5]
            mean_fit = float(np.mean(valid_fits)) if valid_fits else 1e6
            elapsed_gen = time.perf_counter() - t_gen

            # Mostrar resumen de la generacion en consola
            print(
                f"[Gen {gen:02d}/{n_generations:02d}] "
                f" Evaluados: {len(invalid_ind):02d} ({elapsed_gen:5.1f}s) | "
                f"Mejor Fit: {best_fit:8.4f} | "
                f"ATE: {best_info['ate_rmse']:7.4f}m | "
                f"RPE_t: {best_info['rpe_trans_rmse']:6.4f}m | "
                f"RPE_r: {best_info['rpe_rot_rmse']:6.4f}deg | "
                f"Cov: {best_info['coverage']*100:5.1f}%"
            )

            # Guardar Checkpoint JSON del mejor individuo hasta ahora
            checkpoint_data = {
                "generation": gen,
                "best_fitness": best_fit,
                "metrics": best_info,
                "best_params": best_params
            }
            with open(best_json_path, "w", encoding="utf-8") as jf:
                json.dump(checkpoint_data, jf, indent=4)

            # Guardar fila en el CSV
            with open(csv_log_path, mode="a", newline="", encoding="utf-8") as cf:
                writer = csv.writer(cf)
                writer.writerow([
                    gen, round(best_fit, 5),
                    round(best_info["ate_rmse"], 5) if best_info["ate_rmse"] is not None else None,
                    round(best_info["rpe_trans_rmse"], 5) if best_info["rpe_trans_rmse"] is not None else None,
                    round(best_info["rpe_rot_rmse"], 5) if best_info["rpe_rot_rmse"] is not None else None,
                    round(best_info["coverage"], 4),
                    round(mean_fit, 5),
                    round(elapsed_gen, 2)
                ])

            if gen == n_generations:
                break

            # --- Crear siguiente generacion (Seleccion + Cruce + Mutacion + Elitismo) ---
            # 1. Guardar los mejores `elite_size` individuos intactos
            elites = [toolbox.clone(ind) for ind in tools.selBest(pop, elite_size)]

            # 2. Seleccionar padres para el resto de la poblacion
            offspring = toolbox.select(pop, len(pop) - elite_size)
            offspring = [toolbox.clone(ind) for ind in offspring]

            # 3. Aplicar Cruce (SBX)
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < cx_prob:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # 4. Aplicar Mutacion Polinomial
            for mutant in offspring:
                if random.random() < mut_prob:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # 5. Nueva poblacion = Elites + Hijos
            pop[:] = elites + offspring

    print("\n" + "=" * 70)
    print(" OPTIMIZACION FINALIZADA ")
    print("=" * 70)
    print(f"Mejor Fitness : {hof[0].fitness.values[0]:.4f}")
    print(f"Mejor ATE     : {hof[0].metrics['ate_rmse']:.4f} m")
    print(f"Mejor RPE_t   : {hof[0].metrics['rpe_trans_rmse']:.4f} m")
    print(f"Mejor RPE_r   : {hof[0].metrics['rpe_rot_rmse']:.4f} deg")
    print(f"Cobertura     : {hof[0].metrics['coverage']*100:.1f}%")
    print("\nMejores Hiperparametros encontrados (guardados en best_vo_params.json):")
    for k, v in decode_individual(hof[0]).items():
        print(f"  {k:20s}: {v}")


if __name__ == "__main__":
    # Obligatorio en Windows para multiprocessing
    mp.freeze_support()

    BASE_DIR = Path(__file__).parent

    # Ajusta estos 3 nombres a las carpetas exactas que pusiste en evaluator.py:
    SEQ_DIR       = BASE_DIR / "07"
    DISP_DIR      = BASE_DIR / "disparidad"  # <-- Pon aquí el nombre real de tu carpeta de .npy
    GT_POSES_FILE = BASE_DIR / "07.txt"          # <-- O BASE_DIR / "poses" / "07.txt" según dónde lo tengas

    # Numero de procesos en paralelo (4 es ideal para no saturar la RAM con 1100 frames)
    NUM_WORKERS = min(4, max(1, (os.cpu_count() or 4) // 2))

    run_genetic_algorithm(
        seq_dir=SEQ_DIR,
        disp_dir=DISP_DIR,
        gt_poses_file=GT_POSES_FILE,
        start_frame=0,
        num_frames=1100,
        pop_size=20,        # 20 individuos por generacion
        n_generations=15,   # 15 generaciones
        cx_prob=0.75,       # 75% probabilidad de cruce
        mut_prob=0.35,      # 35% probabilidad de mutacion
        elite_size=2,       # Los 2 mejores pasan intactos a la siguiente generacion
        n_workers=NUM_WORKERS,
        seed=42
    )