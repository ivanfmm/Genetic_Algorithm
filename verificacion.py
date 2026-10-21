import numpy as np

def load_kitti_calib(calib_path):
    calib = {}
    with open(calib_path, 'r') as f:
        for line in f:
            if ':' in line:
                k, v = line.split(':', 1)
                calib[k.strip()] = np.fromstring(v, sep=' ').reshape(3, 4)
    
    # P2 y P3 son las cámaras izquierda y derecha a color
    P2, P3 = calib['P2'], calib['P3']
    K = P2[:3, :3].copy()
    fx, cx, cy = K[0, 0], K[0, 2], K[1, 2]
    
    # Baseline en milímetros (~537.15 mm en KITTI)
    baseline_mm = ((P2[0, 3] - P3[0, 3]) / fx) * 1000.0
    
    Q = np.array([
        [1.0, 0.0,                 0.0, -cx],
        [0.0, 1.0,                 0.0, -cy],
        [0.0, 0.0,                 0.0,  fx],
        [0.0, 0.0,  1.0 / baseline_mm, 0.0]
    ], dtype=np.float64)
    
    return K, Q, baseline_mm

# Prueba rápida con tu primer archivo .npy
K, Q, b_mm = load_kitti_calib("07/calib.txt")
disp = np.load("disparidad/000000_disp.npy")

print(f"Shape disp: {disp.shape}, dtype: {disp.dtype}")
print(f"Disparidad min: {disp.min():.2f} px, max: {disp.max():.2f} px")
print(f"Baseline: {b_mm:.2f} mm")

# Profundidad Z (en metros) en la parte baja de la imagen (el piso frente al auto)
disp_piso = disp[300:360, 500:700]
z_metros = (K[0, 0] * b_mm / disp_piso) / 1000.0
print(f"Distancia al piso frente al auto: ~{np.median(z_metros):.2f} metros")