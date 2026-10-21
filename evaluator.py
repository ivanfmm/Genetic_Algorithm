import time
from pathlib import Path
import cv2
import numpy as np

# Importar tu clase limpia desde VO.py
from VO import VisualOdometry, DEFAULT_VO_PARAMS

# Importar modulos internos de evo
from evo.core.trajectory import PoseTrajectory3D
from evo.core import sync
from evo.core import metrics


# ==================================================================
#  1. Funciones de carga de KITTI (Calibracion, Ground Truth y Cache)
# ==================================================================

def load_kitti_calib(calib_path):
    """Lee calib.txt (camaras a color P2 y P3) y devuelve K (3x3) y Q (4x4 en mm)."""
    calib = {}
    with open(calib_path, 'r') as f:
        for line in f:
            if ':' in line:
                k, v = line.split(':', 1)
                calib[k.strip()] = np.fromstring(v, sep=' ').reshape(3, 4)

    P2, P3 = calib['P2'], calib['P3']
    K = P2[:3, :3].copy()
    fx, cx, cy = K[0, 0], K[0, 2], K[1, 2]
    baseline_mm = ((P2[0, 3] - P3[0, 3]) / fx) * 1000.0

    Q = np.array([
        [1.0, 0.0,                 0.0, -cx],
        [0.0, 1.0,                 0.0, -cy],
        [0.0, 0.0,                 0.0,  fx],
        [0.0, 0.0,  1.0 / baseline_mm, 0.0]
    ], dtype=np.float64)

    return K, Q


def load_kitti_gt_poses(poses_path, max_frames=None):
    """
    Lee el archivo de Ground Truth de KITTI (ej. 07.txt) donde cada linea
    tiene 12 flotantes (matriz 3x4 en metros) y devuelve lista de matrices 4x4.
    """
    gt_poses = []
    with open(poses_path, 'r') as f:
        for idx, line in enumerate(f):
            if max_frames is not None and idx >= max_frames:
                break
            vals = np.fromstring(line, sep=' ')
            if len(vals) == 12:
                T = np.eye(4, dtype=np.float64)
                T[:3, :4] = vals.reshape(3, 4)
                gt_poses.append(T)
    return gt_poses


class KITTIDatasetCache:
    """
    Carga en RAM un segmento de imagenes (en escala de grises) y mapas de
    disparidad (.npy) una sola vez para que el Algoritmo Genetico no lea
    del disco duro en cada evaluacion.
    """
    def __init__(self, seq_dir, disp_dir, gt_poses_file, start_frame=0, num_frames=300):
        seq_dir  = Path(seq_dir)
        disp_dir = Path(disp_dir)

        self.K, self.Q = load_kitti_calib(seq_dir / "calib.txt")
        all_gt = load_kitti_gt_poses(gt_poses_file, max_frames=start_frame + num_frames)

        # Seleccionar carpeta de imagenes izquierdas (image_2 para color, image_0 para gris)
        img_folder = seq_dir / "image_2"
        if not img_folder.exists():
            img_folder = seq_dir / "image_0"

        self.frames_gray = []
        self.disparities = []
        self.gt_poses    = []
        self.frame_ids   = []

        end_frame = min(start_frame + num_frames, len(all_gt))
        print(f"[Cache] Cargando frames {start_frame} a {end_frame - 1} en RAM...")

        for fid in range(start_frame, end_frame):
            img_path = img_folder / f"{fid:06d}.png"
            # Soporta tanto 000000_disp.npy como 000000.npy
            disp_path = disp_dir / f"{fid:06d}_disp.npy"
            if not disp_path.exists():
                disp_path = disp_dir / f"{fid:06d}.npy"

            if not img_path.exists() or not disp_path.exists():
                raise FileNotFoundError(f"Falta imagen o disparidad para frame {fid:06d}:\n"
                                        f"  img:  {img_path}\n  disp: {disp_path}")

            # Cargamos directamente en escala de grises (uint8) para ahorrar 3x RAM y tiempo en VO
            gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            # Cargamos disparidad en float16 en RAM (VO._deproject lo pasa a float32 al vuelo)
            disp = np.load(str(disp_path)).astype(np.float16)

            self.frames_gray.append(gray)
            self.disparities.append(disp)
            self.gt_poses.append(all_gt[fid])
            self.frame_ids.append(fid)

        # Normalizar las poses GT respecto al primer frame del segmento
        T0_inv = np.linalg.inv(self.gt_poses[0])
        self.gt_poses = [T0_inv @ T for T in self.gt_poses]
        self.total_frames = len(self.frame_ids)
        print(f"[Cache] Listos {self.total_frames} frames en memoria.")


# ==================================================================
#  2. Funcion de Evaluacion (Fitness) con evo: ATE + RPE + Cobertura
# ==================================================================

def evaluate_vo_params(params, dataset_cache,
                       w_ate=1.0, w_rpe_trans=1.0, w_rpe_rot=0.1,
                       min_coverage=0.50, penalty_value=1e6):
    """
    Ejecuta VisualOdometry con el diccionario `params` sobre `dataset_cache`
    y calcula el Fitness usando la libreria `evo`.

    Retorna:
        fitness (float): Valor a minimizar por DEAP.
        info (dict): Metricas desglosadas (ATE, RPE_trans, RPE_rot, cobertura).
    """
    vo = VisualOdometry(
        Q=dataset_cache.Q,
        K=dataset_cache.K,
        params=params,
        verbose=False
    )

    # 1. Ejecutar Visual Odometry en todos los frames del segmento
    for idx in range(dataset_cache.total_frames):
        fid  = dataset_cache.frame_ids[idx]
        gray = dataset_cache.frames_gray[idx]
        disp = dataset_cache.disparities[idx]
        vo.update(gray, disp, frame_idx=fid)

    est_poses_dict = vo.get_all_poses()
    num_est = len(est_poses_dict)
    coverage = num_est / float(dataset_cache.total_frames)

    # 2. Restriccion dura (Muerte subita si pierde demasiados frames)
    if coverage < min_coverage or num_est < 10:
        return penalty_value, {
            "ate_rmse": None,
            "rpe_trans_rmse": None,
            "rpe_rot_rmse": None,
            "coverage": coverage,
            "valid": False
        }

    # 3. Construir objetos PoseTrajectory3D de evo en METROS
    est_timestamps = []
    est_se3_list   = []
    for fid in sorted(est_poses_dict.keys()):
        T_mm = est_poses_dict[fid].copy()
        T_m  = T_mm.astype(np.float64)
        T_m[:3, 3] /= 1000.0  # Convertir traslacion de mm a metros
        est_timestamps.append(float(fid))
        est_se3_list.append(T_m)

    gt_timestamps = [float(fid) for fid in dataset_cache.frame_ids]
    gt_se3_list   = dataset_cache.gt_poses

    try:
        traj_est = PoseTrajectory3D(poses_se3=est_se3_list, timestamps=np.array(est_timestamps))
        traj_gt  = PoseTrajectory3D(poses_se3=gt_se3_list,  timestamps=np.array(gt_timestamps))

        # Sincronizar por frame_idx (descarta en GT los frames donde VO fallo)
        traj_gt_sync, traj_est_sync = sync.associate_trajectories(traj_gt, traj_est, max_diff=0.1)

        # Alinear origen/orientacion con Umeyama (SIN corregir escala porque es estereo)
        traj_est_aligned = PoseTrajectory3D(
            poses_se3=[p.copy() for p in traj_est_sync.poses_se3],
            timestamps=traj_est_sync.timestamps.copy()
        )
        traj_est_aligned.align(traj_gt_sync, correct_scale=False)

        # 4. Calcular ATE (Absolute Trajectory Error - Traslacion en metros)
        ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
        ape_metric.process_data((traj_gt_sync, traj_est_aligned))
        ate_rmse = float(ape_metric.get_statistic(metrics.StatisticsType.rmse))

        # 5. Calcular RPE (Relative Pose Error frame a frame: Traslacion en m y Rotacion en deg)
        rpe_trans_metric = metrics.RPE(
            metrics.PoseRelation.translation_part,
            delta=1.0, delta_unit=metrics.Unit.frames, all_pairs=False
        )
        rpe_trans_metric.process_data((traj_gt_sync, traj_est_sync))
        rpe_trans_rmse = float(rpe_trans_metric.get_statistic(metrics.StatisticsType.rmse))

        rpe_rot_metric = metrics.RPE(
            metrics.PoseRelation.rotation_angle_deg,
            delta=1.0, delta_unit=metrics.Unit.frames, all_pairs=False
        )
        rpe_rot_metric.process_data((traj_gt_sync, traj_est_sync))
        rpe_rot_rmse = float(rpe_rot_metric.get_statistic(metrics.StatisticsType.rmse))

    except Exception as e:
        # En caso de trayectorias degeneradas (ej. matriz singular en Umeyama)
        return penalty_value, {
            "ate_rmse": None, "rpe_trans_rmse": None, "rpe_rot_rmse": None,
            "coverage": coverage, "valid": False, "error": str(e)
        }

    if not np.isfinite(ate_rmse) or not np.isfinite(rpe_trans_rmse) or not np.isfinite(rpe_rot_rmse):
        return penalty_value, {
            "ate_rmse": None, "rpe_trans_rmse": None, "rpe_rot_rmse": None,
            "coverage": coverage, "valid": False
        }

    # 6. Funcion de aptitud combinada con penalizacion cuadratica por cobertura
    raw_error = (w_ate * ate_rmse) + (w_rpe_trans * rpe_trans_rmse) + (w_rpe_rot * rpe_rot_rmse)
    fitness = raw_error / (coverage ** 2)

    return float(fitness), {
        "ate_rmse": ate_rmse,
        "rpe_trans_rmse": rpe_trans_rmse,
        "rpe_rot_rmse": rpe_rot_rmse,
        "coverage": coverage,
        "valid": True
    }


# ==================================================================
#  3. Prueba rapida del Evaluador (Linea Base antes del AG)
# ==================================================================
if __name__ == "__main__":
    # Ajusta estas 3 rutas a tu estructura local de carpetas:
    BASE_DIR = Path(__file__).parent

    SEQ_DIR       = BASE_DIR / "07"
    DISP_DIR      = BASE_DIR / "disparidad"  # <-- Pon aquí el nombre real de tu carpeta de .npy
    GT_POSES_FILE = BASE_DIR / "07.txt"          # <-- O BASE_DIR / "poses" / "07.txt" según dónde lo tengas


    # Puedes probar primero con 200 o 300 frames (o todos los que hayas precalculado)
    cache = KITTIDatasetCache(
        seq_dir=SEQ_DIR,
        disp_dir=DISP_DIR,
        gt_poses_file=GT_POSES_FILE,
        start_frame=0,
        num_frames=1100
    )

    t0 = time.perf_counter()
    fitness, info = evaluate_vo_params(DEFAULT_VO_PARAMS, cache)
    elapsed = time.perf_counter() - t0

    print("\n=== Resultado Linea Base (DEFAULT_VO_PARAMS) ===")
    print(f"Tiempo de evaluacion : {elapsed:.2f} s")
    print(f"Cobertura de frames  : {info['coverage']*100:.1f}%")
    print(f"ATE RMSE (m)         : {info['ate_rmse']:.4f} m")
    print(f"RPE Trans RMSE (m)   : {info['rpe_trans_rmse']:.4f} m")
    print(f"RPE Rot RMSE (deg)   : {info['rpe_rot_rmse']:.4f} deg")
    print(f"FITNESS TOTAL        : {fitness:.4f}")