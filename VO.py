import cv2
import numpy as np
import faiss


LOOP_CHECK_EVERY  = 1     # buscar loop closure en CADA frame
LC_KF_MOV_MM    = 50.0    # mm que debe avanzar el carro desde el ultimo keyframe agregado
LC_KF_ROT_DEG   = 5.0     # o grados que debe girar
# Loop closure - filtro de calidad FAISS
LC_HAMMING_MAX  = 80      # distancia Hamming maxima para contar voto valido \ era 64
# ==================================================================
#  Constantes de loop closure y PGO
# ==================================================================
MIN_LOOP_SEP      = 80    # frames minimos de separacion para loop closure \eran 15

# ==================================================================
#  Hiperparametros por defecto (adaptados a escala KITTI en mm)
# ==================================================================
DEFAULT_VO_PARAMS = {
    # ORB features
    "orb_nfeatures":     1500,    # int   [500, 3000]
    "orb_scaleFactor":   1.2,     # float [1.1, 1.5]
    "orb_nlevels":       6,       # int   [4, 10]
    "orb_edgeThreshold": 19,      # int   [15, 35]
    "orb_patchSize":     19,      # int   [15, 35] (se recomienda igualar a edgeThreshold)
    "orb_fastThreshold": 10,      # int   [5, 25]

    # Matching y deproyeccion
    "lowe_ratio":        0.75,    # float [0.60, 0.90]
    "disp_min":          2.0,     # float [1.0, 10.0]

    # PnP RANSAC
    "pnp_iterations":    200,     # int   [100, 500]
    "pnp_reproj_err":    2.0,     # float [0.5, 4.0]
    "min_pnp_matches":   12,      # int   [8, 40]
    "min_pnp_inliers":   12,      # int   [8, 40]

    # Filtro de saltos y recuperacion (escala auto KITTI en mm)
    "jump_max_trans":    2500.0,  # float [1500.0, 4000.0] mm/frame
    "jump_max_rot":      20.0,    # float [5.0, 30.0] deg/frame
    "max_fallos":        5,       # int   [2, 10]

    # Restriccion planar (piso en escala KITTI en mm)
    "use_planar":        True,
    "floor_band":        0.35,    # float [0.20, 0.45]
    "floor_z_max":       15000.0, # float [8000.0, 20000.0] mm
    "floor_ransac_mm":   40.0,    # float [15.0, 80.0] mm
    "floor_min_ratio":   0.4,     # float [0.25, 0.60]
    "floor_max_tries":   30,

    # Modulos opcionales (apagados para optimizar VO pura)
    "use_icp":           False,
    "use_loop_closure":  False,
}


class VisualOdometry:
    def __init__(self, Q, K, inclinacion_deg=None, params=None, verbose=False):
        """
        Q : matriz de reproyeccion disparidad->3D (4x4, en mm)
        K : matriz intrinseca del lente izquierdo rectificado (3x3)
        inclinacion_deg : angulo de la camara respecto al piso (+ = hacia arriba)
        params : dict opcional con hiperparametros a sobrescribir
        verbose : bool, imprime mensajes de depuracion si es True
        """
        self.Q = Q
        self.K = K
        self.verbose = verbose

        # Combinar parametros por defecto con los recibidos del AG
        cfg = DEFAULT_VO_PARAMS.copy()
        if params is not None:
            cfg.update(params)
        self.cfg = cfg

        # Guardar atributos rapidos para el loop de update()
        self.lowe_ratio       = float(cfg["lowe_ratio"])
        self.disp_min         = float(cfg["disp_min"])
        self.pnp_iterations   = int(round(cfg["pnp_iterations"]))
        self.pnp_reproj_err   = float(cfg["pnp_reproj_err"])
        self.min_pnp_matches  = int(round(cfg["min_pnp_matches"]))
        self.min_pnp_inliers  = int(round(cfg["min_pnp_inliers"]))
        self.jump_max_trans   = float(cfg["jump_max_trans"])
        self.jump_max_rot     = float(cfg["jump_max_rot"])
        self.max_fallos       = int(round(cfg["max_fallos"]))

        self.use_planar       = bool(cfg["use_planar"])
        self.floor_band       = float(cfg["floor_band"])
        self.floor_z_max      = float(cfg["floor_z_max"])
        self.floor_ransac_mm  = float(cfg["floor_ransac_mm"])
        self.floor_min_ratio  = float(cfg["floor_min_ratio"])
        self.floor_max_tries  = int(round(cfg["floor_max_tries"]))

        self.use_icp          = bool(cfg["use_icp"])
        self.use_loop_closure = bool(cfg["use_loop_closure"])

        # Asegurar tipos enteros y consistencia para OpenCV ORB
        patch_sz = max(2, int(round(cfg["orb_patchSize"])))
        edge_th  = max(patch_sz, int(round(cfg["orb_edgeThreshold"])))

        self.orb = cv2.ORB_create(
            nfeatures     = int(round(cfg["orb_nfeatures"])),
            scaleFactor   = float(cfg["orb_scaleFactor"]),
            nlevels       = int(round(cfg["orb_nlevels"])),
            edgeThreshold = edge_th,
            patchSize     = patch_sz,
            fastThreshold = int(round(cfg["orb_fastThreshold"])),
        )
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

        self.prev_frame = None
        self.prev_kp    = None
        self.prev_des   = None
        self.prev_disp  = None

        self.pose = np.eye(4)
        self.all_poses  = {}
        self.seq_edges  = []
        self.loop_edges = []
        self._kf_data   = {}

        self._loop_detector  = _LoopDetector() if self.use_loop_closure else None
        self._prev_idx       = None
        self._internal_cnt   = 0
        self._last_pgo_frame = 0
        self._fallos         = 0
        self._last_lc_pose   = None

        self.A            = None
        self.floor_info   = None
        self._floor_tries = 0
        if self.use_planar and inclinacion_deg is not None:
            th = np.radians(inclinacion_deg)
            self._set_floor(np.array([0.0, np.cos(th), -np.sin(th)]), {'origen': 'parametro'})

    def update(self, frame, disp, frame_idx=None):
        if frame_idx is None:
            frame_idx = self._internal_cnt
        self._internal_cnt += 1

        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        disp = self._ensure_2d(disp)

        # Detectar el piso (solo hasta encontrarlo)
        if self.use_planar and self.A is None and self._floor_tries < self.floor_max_tries:
            self._floor_tries += 1
            self._detect_floor(disp, frame_idx)

        kp, des = self.orb.detectAndCompute(gray, None)

        # Solo guardar _kf_data si loop closure esta activo (ahorra RAM y tiempo en el AG)
        if self.use_loop_closure:
            if kp:
                uv_kp    = np.float32([k.pt for k in kp])
                pts3d_kp = self._deproject(uv_kp, disp)
            else:
                uv_kp    = np.zeros((0, 2), dtype=np.float32)
                pts3d_kp = np.zeros((0, 3), dtype=np.float32)
            self._kf_data[frame_idx] = {'des': des, 'pts3d': pts3d_kp, 'uv': uv_kp}

        self.all_poses[frame_idx] = self.pose.copy()

        if self.prev_frame is None or self.prev_des is None:
            self._save_state(gray, kp, des, disp)
            self._prev_idx = frame_idx
            self._add_loop_keyframe(frame_idx, des)
            return None

        if des is None or len(des) < 8:
            if self.verbose:
                print(f"Frame {frame_idx} failed: des < 8")
            self._fallo(gray, kp, des, disp, frame_idx)
            return None

        raw = self.matcher.knnMatch(self.prev_des, des, k=2)
        matches = [
            pair[0] for pair in raw
            if len(pair) == 2 and pair[0].distance < self.lowe_ratio * pair[1].distance
        ]
        if len(matches) < self.min_pnp_matches:
            if self.verbose:
                print(f"Frame {frame_idx} failed: matches < min_pnp_matches")
            self._fallo(gray, kp, des, disp, frame_idx)
            return None

        pts_prev_2d = np.float32([self.prev_kp[m.queryIdx].pt for m in matches])
        pts_curr_2d = np.float32([kp[m.trainIdx].pt           for m in matches])

        pts3d = self._deproject(pts_prev_2d, self.prev_disp)
        valid = pts3d[:, 2] > 0
        if valid.sum() < self.min_pnp_inliers:
            if self.verbose:
                print(f"Frame {frame_idx} failed: valid 3D < min_pnp_inliers")
            self._fallo(gray, kp, des, disp, frame_idx)
            return None

        pts3d       = pts3d[valid]
        pts_curr_2d = pts_curr_2d[valid]

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts3d, pts_curr_2d, self.K, None,
            iterationsCount=self.pnp_iterations,
            reprojectionError=self.pnp_reproj_err,
        )

        if not success or inliers is None or len(inliers) < self.min_pnp_inliers:
            if self.verbose:
                print(f"Frame {frame_idx} failed: PnP inliers < min_pnp_inliers")
            self._fallo(gray, kp, des, disp, frame_idx)
            return None

        R, _ = cv2.Rodrigues(rvec)
        tvec_raw  = tvec.flatten()
        tvec_mag  = float(np.linalg.norm(tvec_raw))
        angle_deg = float(np.degrees(np.linalg.norm(rvec.flatten())))

        if tvec_mag > self.jump_max_trans or angle_deg > self.jump_max_rot:
            if self.verbose:
                print(f"  [VO] SALTO frame {frame_idx}: |t|={tvec_mag:.0f}mm rot={angle_deg:.1f}deg")
            self._fallo(gray, kp, des, disp, frame_idx)
            return None

        self._fallos = 0

        T = np.eye(4)
        T[:3, :3] = R
        T[:3,  3] = tvec_raw
        if self.use_icp:
            T = self._icp_refine(T, self.prev_disp, disp)
        if self.use_planar and self.A is not None:
            T = np.linalg.inv(self._to_planar(np.linalg.inv(T)))

        self.pose = self.pose @ np.linalg.inv(T)
        self.all_poses[frame_idx] = self.pose.copy()

        if self._prev_idx is not None and self._prev_idx in self.all_poses:
            T_rel = np.linalg.inv(self.all_poses[self._prev_idx]) @ self.pose
            self.seq_edges.append((self._prev_idx, frame_idx, T_rel))

        self._add_loop_keyframe(frame_idx, des)

        if self.use_loop_closure and self._internal_cnt % LOOP_CHECK_EVERY == 0:
            self._check_loop_closure(frame_idx, kp, des, disp)

        self._save_state(gray, kp, des, disp)
        self._prev_idx = frame_idx
        return R, tvec.flatten()

    def _detect_floor(self, disp, frame_idx):
        h, w = disp.shape
        y0 = int(h * (1 - self.floor_band))
        d  = disp[y0::4, ::4].astype(np.float32).ravel()
        vs, us = np.mgrid[y0:h:4, 0:w:4]
        uvd1 = np.stack([us.ravel(), vs.ravel(), d, np.ones_like(d)], axis=1)
        uvd1 = uvd1[d > self.disp_min]
        if len(uvd1) < 500:
            return False
        P = (self.Q @ uvd1.T).T
        pts = P[:, :3] / P[:, 3:4]
        pts = pts[(pts[:, 2] > 100) & (pts[:, 2] < self.floor_z_max)]
        if len(pts) < 500:
            return False

        rng = np.random.default_rng(0)
        best = None
        for _ in range(200):
            p = pts[rng.choice(len(pts), 3, replace=False)]
            n = np.cross(p[1] - p[0], p[2] - p[0])
            if np.linalg.norm(n) < 1e-6:
                continue
            n /= np.linalg.norm(n)
            inl = np.abs((pts - p[0]) @ n) < self.floor_ransac_mm
            if best is None or inl.sum() > best.sum():
                best = inl
        if best is None or best.mean() < self.floor_min_ratio:
            return False

        q = pts[best]
        c = q.mean(axis=0)
        n = np.linalg.svd(q - c)[2][2]
        if n[1] < 0:
            n = -n
        if n[1] < 0.7:
            return False

        self._set_floor(n, {'origen': f'piso detectado en frame {frame_idx}',
                            'altura_mm': float(abs(n @ c))})
        return True

    def _set_floor(self, n, info):
        n = n / np.linalg.norm(n)
        y_f = n
        z_f = np.array([0.0, 0.0, 1.0]) - n[2] * n
        z_f /= np.linalg.norm(z_f)
        x_f = np.cross(y_f, z_f)
        self.A = np.stack([x_f, y_f, z_f])

        info['inclinacion_deg'] = float(np.degrees(np.arcsin(-n[2])))
        info['alabeo_deg']      = float(np.degrees(np.arcsin(n[0])))
        self.floor_info = info
        if self.verbose:
            print(f"  [VO] Movimiento plano ({info['origen']}): inclinacion "
                  f"{info['inclinacion_deg']:+.1f} deg, alabeo {info['alabeo_deg']:+.1f} deg")

    def _to_planar(self, M):
        A = self.A
        R = A @ M[:3, :3] @ A.T
        t = A @ M[:3, 3]
        yaw = np.arctan2(R[0, 2] - R[2, 0], R[0, 0] + R[2, 2])
        c, s = np.cos(yaw), np.sin(yaw)
        R_p = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        t_p = np.array([t[0], 0.0, t[2]])
        out = np.eye(4)
        out[:3, :3] = A.T @ R_p @ A
        out[:3, 3]  = A.T @ t_p
        return out

    def get_all_poses(self):
        return {k: v.copy() for k, v in self.all_poses.items()}

    def _save_state(self, gray, kp, des, disp):
        self.prev_frame = gray
        self.prev_kp    = kp
        self.prev_des   = des
        self.prev_disp  = disp

    def _fallo(self, gray, kp, des, disp, frame_idx):
        self._fallos += 1
        if self._fallos >= self.max_fallos:
            if self.verbose:
                print(f"  [VO] {self.max_fallos} fallos seguidos: nueva referencia en frame {frame_idx}")
            self._fallos = 0
            self._save_state(gray, kp, des, disp)
            self._prev_idx = frame_idx
        else:
            self.all_poses.pop(frame_idx, None)

    def _add_loop_keyframe(self, frame_idx, des):
        if not self.use_loop_closure or des is None or len(des) == 0:
            return
        if self._last_lc_pose is not None:
            rel = np.linalg.inv(self._last_lc_pose) @ self.pose
            mov = np.linalg.norm(rel[:3, 3])
            ang = np.degrees(np.linalg.norm(cv2.Rodrigues(rel[:3, :3])[0]))
            if mov < LC_KF_MOV_MM and ang < LC_KF_ROT_DEG:
                return
        self._loop_detector.add(frame_idx, des)
        self._last_lc_pose = self.pose.copy()

    @staticmethod
    def _ensure_2d(disp):
        disp = np.squeeze(disp)
        if disp.ndim == 2:
            return disp
        if disp.ndim == 1:
            raise ValueError(f"_ensure_2d: disp llego como 1D con shape {disp.shape}.")
        while disp.ndim > 2:
            disp = disp[0]
        return disp

    def _deproject(self, pts2d, disp):
        disp = self._ensure_2d(disp)
        uv   = pts2d.astype(int)
        us   = uv[:, 0]
        vs   = uv[:, 1]
        oob  = (
            (us < 0) | (us >= disp.shape[1]) |
            (vs < 0) | (vs >= disp.shape[0])
        )
        us_c = np.clip(us, 0, disp.shape[1] - 1)
        vs_c = np.clip(vs, 0, disp.shape[0] - 1)
        ds   = disp[vs_c, us_c].astype(np.float32)
        ds[oob] = 0.0
        ones  = np.ones(len(uv), dtype=np.float32)
        pts_h = np.stack(
            [us.astype(np.float32), vs.astype(np.float32), ds, ones], axis=1
        )
        vecs  = (self.Q @ pts_h.T).T
        w     = vecs[:, 3:4]
        pts3d = vecs[:, :3] / np.where(np.abs(w) > 1e-6, w, 1.0)
        pts3d[ds < self.disp_min] = 0.0
        return pts3d.astype(np.float32)





class _LoopDetector:
    """
    Mantiene un indice FAISS de descriptores ORB binarios (256 bits = 32 bytes).
    Usa distancia Hamming para encontrar descriptores similares entre el frame
    actual y frames almacenados con separacion >= MIN_LOOP_SEP.
    """

    def __init__(self):
        self.index     = faiss.IndexBinaryFlat(256)
        self._frame_of = []   # posicion en index -> frame_idx

    def add(self, frame_idx, descriptors):
        """Agrega todos los descriptores de un frame al indice."""
        if descriptors is None or len(descriptors) == 0:
            return
        des = np.ascontiguousarray(descriptors, dtype=np.uint8)
        self.index.add(des)
        self._frame_of.extend([frame_idx] * len(des))

    def query(self, descriptors, current_idx):
        """
        Busca el frame pasado mas similar al actual por voto de descriptores.

        Retorna (best_frame_idx, vote_count) o (None, 0) si no hay candidato
        con suficiente separacion.
        """
        if descriptors is None or len(descriptors) == 0:
            return None, 0
        if self.index.ntotal == 0:
            return None, 0

        des = np.ascontiguousarray(descriptors, dtype=np.uint8)
        D, I = self.index.search(des, 1)   # D: distancias Hamming, I: indices

        votes = {}
        for (dist,), (nn_id,) in zip(D, I):
            if nn_id < 0:
                continue
            if int(dist) > LC_HAMMING_MAX:
                continue   # match de baja calidad: distancia > 64/256 bits
            fid = self._frame_of[nn_id]
            if current_idx - fid < MIN_LOOP_SEP:
                continue
            votes[fid] = votes.get(fid, 0) + 1

        if not votes:
            return None, 0
        best = max(votes, key=votes.get)
        return best, votes[best]