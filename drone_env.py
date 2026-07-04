"""
Drone Agricole Hexacoptère

"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import os

# ─────────────────────── Constantes physiques ─────────────────────────────────
MASSE_INITIALE  = 15.0
MASSE_MIN       = 10.0
GRAVITE         = 9.81
DT              = 1.0 / 240.0
ALTITUDE_TRAVAIL= 15.0
FORCE_MAX       = 400.0         # N — augmenté pour compenser masse réelle URDF
COUPLE_MAX      = 5.0           # Nm — couple réduit pour stabilité maximale
DEBIT_PULV      = 0.03          # kg/s
MAX_INCLINAISON = 1.4           # rad (~80°) — tolérant pour éviter faux crashes

# ─────────────────────── Géométrie du champ ───────────────────────────────────
RANGEES_Y   = [-4.5, -1.5, 1.5, 4.5]
X_DEBUT     = -20.0
X_FIN       =  20.0

# ─────────────────────── Paramètres RL ───────────────────────────────────────
MAX_STEPS       = 4000
TOLÉRANCE_Y     = 0.5
TOLÉRANCE_Z     = 1.0
SEUIL_CRASH_Z   = 0.25

# ─────────────────────── Récompenses ─────────────────────────────────────────
R_SURVIE        =  0.02   # bonus par step en l'air (signal dense précoce)
R_ALTITUDE_OK   =  0.10
R_ALIGNEMENT_Y  =  0.15
R_AVANCEMENT    =  0.20
R_PULVERISATION =  0.05
R_FIN_RANGEE    = 10.0
R_MISSION_FINIE = 50.0
P_CRASH         = -30.0
P_DIVERGENCE    =  -5.0
P_INCLINAISON   =  -0.10


class DroneAgricoleEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None, urdf_path=None):
        super().__init__()
        self.render_mode = render_mode
        self.urdf_path   = urdf_path or "models/agri_hexacopter.urdf"
        self._physics_client = None

        obs_low = np.array([
            -30., -10., 0.,
            -np.pi, -np.pi, -np.pi,
            -10., -10., -10.,
            -5., -5., -5.,
            MASSE_MIN, 0., -45., -10., -20., 0.
        ], dtype=np.float32)
        obs_high = np.array([
             30., 10., 30.,
             np.pi, np.pi, np.pi,
             10., 10., 10.,
             5., 5., 5.,
             MASSE_INITIALE, 3., 45., 10., 20., 1.
        ], dtype=np.float32)

        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.full(4, -1., dtype=np.float32),
            high=np.full(4,  1., dtype=np.float32),
        )

        self._drone_id      = None
        self._step_count    = 0
        self._masse         = MASSE_INITIALE
        self._rangee_idx    = 0
        self._dir_forward   = True
        self._rangees_faites= 0
        self._pulv_active   = False
        self._prev_x        = X_DEBUT
        self._stats         = {}

    # ── RESET ─────────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self._physics_client is not None:
            try:
                p.disconnect(self._physics_client)
            except Exception:
                pass

        cid = p.connect(p.GUI if self.render_mode == "human" else p.DIRECT)
        self._physics_client = cid

        if self.render_mode == "human":
            p.resetDebugVisualizerCamera(
                cameraDistance=34, cameraYaw=38, cameraPitch=-28,
                cameraTargetPosition=[0, 0, 3], physicsClientId=cid
            )
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS,       1, physicsClientId=cid)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI,           0, physicsClientId=cid)
            p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0, physicsClientId=cid)

        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)
        p.setGravity(0, 0, -GRAVITE, physicsClientId=cid)
        p.setTimeStep(DT, physicsClientId=cid)

        # Sol réaliste : terre + bandes herbe
        self._creer_sol(cid)
        self._generer_champ(cid)

        start_pos = [X_DEBUT, RANGEES_Y[0], 2.0]  # départ à 2m (évite crash immédiat)
        if os.path.exists(self.urdf_path):
            self._drone_id = p.loadURDF(self.urdf_path, start_pos, physicsClientId=cid)

            # ── Neutraliser la masse des liens URDF secondaires ────────────
            # prop_1..6, agri_payload, landing_gear, arms_visual n'ont pas de
            # données inertielles dans l'URDF → PyBullet leur assigne mass=1 kg
            # chacun → centre de masse déporté → bascule en 4 steps.
            # Solution : mass=0 sur tous les liens sauf le corps principal (-1).
            n_joints = p.getNumJoints(self._drone_id, physicsClientId=cid)
            for i in range(n_joints):
                p.changeDynamics(
                    self._drone_id, i,
                    mass=0.0,
                    localInertiaDiagonal=[0.0001, 0.0001, 0.0001],
                    linearDamping=0.9,
                    angularDamping=0.99,
                    physicsClientId=cid
                )
            # Corps principal : masse + tenseur d'inertie + amortissement fort
            p.changeDynamics(
                self._drone_id, -1,
                mass=MASSE_INITIALE,
                localInertiaDiagonal=[1.875, 1.875, 3.0],
                linearDamping=0.9,    # frein translation (simule résistance air)
                angularDamping=0.99,  # frein rotation fort (stabilise inclinaison)
                physicsClientId=cid
            )
        else:
            self._drone_id = self._creer_drone_simple(cid, start_pos)

        for i in range(p.getNumJoints(self._drone_id, physicsClientId=cid)):
            p.setJointMotorControl2(self._drone_id, i, p.VELOCITY_CONTROL,
                                    force=0, physicsClientId=cid)

        # État interne
        self._step_count     = 0
        self._masse          = MASSE_INITIALE
        self._rangee_idx     = 0
        self._dir_forward    = True
        self._rangees_faites = 0
        self._pulv_active    = False
        self._prev_x         = X_DEBUT
        self._stats = {"total_pulverise": 0., "rangees_finies": 0,
                       "crash": False, "steps": 0}

        # Préchauffage avec poussée ≈ poids (120 pas pour bien stabiliser
        # les joints continus des props et éviter les oscillations initiales)
        masse_reelle = p.getDynamicsInfo(self._drone_id, -1, physicsClientId=cid)[0]
        poids = masse_reelle * GRAVITE
        for _ in range(120):
            p.applyExternalForce(
                self._drone_id, -1, [0, 0, poids],
                [0, 0, 0], p.LINK_FRAME, physicsClientId=cid
            )
            p.stepSimulation(physicsClientId=cid)

        # Remettre le compteur de steps à zéro après le préchauffage
        self._step_count = 0

        return self._get_obs(), self._get_info()

    # ── STEP ──────────────────────────────────────────────────────────────────
    def step(self, action):
        cid = self._physics_client
        self._step_count += 1

        # Actions → forces/couples
        thrust  = (float(action[0]) + 1.0) / 2.0 * FORCE_MAX * 2.0
        roll_t  = float(action[1]) * COUPLE_MAX
        pitch_t = float(action[2]) * COUPLE_MAX
        yaw_t   = float(action[3]) * COUPLE_MAX

        # Poussée en LINK_FRAME : suit l'axe du corps (comme de vrais rotors)
        # Couples en WORLD_FRAME : contrôle intuitif roll/pitch/yaw
        p.applyExternalForce(
            self._drone_id, -1, [0, 0, thrust],
            [0, 0, 0], p.LINK_FRAME, physicsClientId=cid
        )
        p.applyExternalTorque(
            self._drone_id, -1, [roll_t, pitch_t, yaw_t],
            p.WORLD_FRAME, physicsClientId=cid
        )
        p.stepSimulation(physicsClientId=cid)

        pos, orn     = p.getBasePositionAndOrientation(self._drone_id, physicsClientId=cid)
        lin_vel, _   = p.getBaseVelocity(self._drone_id, physicsClientId=cid)
        euler        = p.getEulerFromQuaternion(orn)
        x, y, z      = pos
        target_y     = RANGEES_Y[self._rangee_idx]
        target_x     = X_FIN if self._dir_forward else X_DEBUT

        # Pulvérisation
        self._pulv_active = (abs(z - ALTITUDE_TRAVAIL) < TOLÉRANCE_Z and
                             abs(y - target_y) < TOLÉRANCE_Y and z > 2.0)
        if self._pulv_active and self._masse > MASSE_MIN:
            self._masse -= DEBIT_PULV * DT
            self._stats["total_pulverise"] += DEBIT_PULV * DT
            p.changeDynamics(self._drone_id, -1, mass=self._masse, physicsClientId=cid)

        # Fin de rangée
        rangee_finie = (self._dir_forward and x >= X_FIN) or \
                       (not self._dir_forward and x <= X_DEBUT)
        if rangee_finie:
            self._rangees_faites += 1
            self._stats["rangees_finies"] = self._rangees_faites
            if self._rangee_idx < 3:
                self._rangee_idx += 1
                self._dir_forward = not self._dir_forward

        # Détection crash (altitude ET bascule)
        # Le délai > 30 évite les faux crashes au tout début de l'épisode
        # quand les joints continus oscillent légèrement après le reset
        crash_altitude = z < SEUIL_CRASH_Z and self._step_count > 60
        crash_bascule  = (abs(euler[0]) > MAX_INCLINAISON or
                          abs(euler[1]) > MAX_INCLINAISON) and self._step_count > 30
        crash          = crash_altitude or crash_bascule
        hors_limites   = abs(x) > 35 or abs(y) > 15 or z > 40

        # ── Récompense ────────────────────────────────────────────────────
        reward = 0.0

        # ── CURRICULUM DE RÉCOMPENSE ──────────────────────────────────────
        # Niveau 0 — Survie en l'air (signal le plus dense, appris en 1er)
        if z > SEUIL_CRASH_Z:
            reward += 0.05

        # Niveau 1 — Monter vers l'altitude de travail
        z_err = abs(z - ALTITUDE_TRAVAIL)
        if z_err < TOLÉRANCE_Z:
            reward += R_ALTITUDE_OK
        else:
            if z < ALTITUDE_TRAVAIL:
                reward += 0.015 * min(z, ALTITUDE_TRAVAIL)  # encourage à monter
            reward -= 0.02 * z_err

        # Niveau 2 — S'aligner sur la rangée (seulement si déjà en l'air)
        y_err = abs(y - target_y)
        if z > 3.0:
            if y_err < TOLÉRANCE_Y:
                reward += R_ALIGNEMENT_Y
            else:
                reward -= 0.015 * y_err

        # Niveau 3 — Progresser sur la rangée
        dx = (x - self._prev_x) if self._dir_forward else (self._prev_x - x)
        if dx > 0 and z > 3.0:
            reward += R_AVANCEMENT * dx
        self._prev_x = x

        # Niveau 4 — Pulvérisation (bonus ×3 pour fortement encourager)
        if self._pulv_active:
            reward += R_PULVERISATION * 3.0

        # Niveau 5 — Fin de rangée et mission complète
        if rangee_finie:
            reward += R_FIN_RANGEE

        # Pénalités
        incli = abs(euler[0]) + abs(euler[1])
        if incli > 0.5:
            reward += P_INCLINAISON * incli

        mission_finie = self._rangees_faites >= 4
        if mission_finie:
            reward += R_MISSION_FINIE
        if crash:
            reward += P_CRASH
            self._stats["crash"] = True
        if hors_limites:
            reward += P_DIVERGENCE

        terminated = crash or mission_finie
        truncated  = self._step_count >= MAX_STEPS or hors_limites
        self._stats["steps"] = self._step_count

        return self._get_obs(), float(reward), terminated, truncated, self._get_info()

    # ── Observation ───────────────────────────────────────────────────────────
    def _get_obs(self):
        cid = self._physics_client
        pos, orn     = p.getBasePositionAndOrientation(self._drone_id, physicsClientId=cid)
        lin_vel, ang = p.getBaseVelocity(self._drone_id, physicsClientId=cid)
        euler        = p.getEulerFromQuaternion(orn)
        x, y, z      = pos
        target_y     = RANGEES_Y[self._rangee_idx]
        target_x     = X_FIN if self._dir_forward else X_DEBUT
        obs = np.array([
            x, y, z,
            euler[0], euler[1], euler[2],
            lin_vel[0], lin_vel[1], lin_vel[2],
            ang[0], ang[1], ang[2],
            self._masse,
            float(self._rangee_idx),
            target_x - x,
            y - target_y,
            z - ALTITUDE_TRAVAIL,
            float(self._pulv_active),
        ], dtype=np.float32)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def _get_info(self):
        return {
            "rangee": self._rangee_idx,
            "rangees_finies": self._rangees_faites,
            "masse": round(self._masse, 3),
            "pulv_active": self._pulv_active,
            "step": self._step_count,
            **self._stats,
        }

    # ═══════════════════════════════════════════════════════════════════════
    # VISUELS AMÉLIORÉS
    # ═══════════════════════════════════════════════════════════════════════

    def _creer_sol(self, cid):
        """Sol réaliste : grande dalle terre brune + bandes vertes entre rangées."""
        # Dalle principale — terre agricole
        vs_sol = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[26., 9., 0.05],
            rgbaColor=[0.45, 0.30, 0.15, 1],   # brun terre
            physicsClientId=cid
        )
        cs_sol = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[26., 9., 0.05],
            physicsClientId=cid
        )
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=cs_sol,
                          baseVisualShapeIndex=vs_sol,
                          basePosition=[0., 0., -0.05], physicsClientId=cid)

        # Bandes de terre labourée sous chaque rangée
        COULEURS_BANDES = [0.38, 0.25, 0.10]
        for ry in RANGEES_Y:
            vs_b = p.createVisualShape(
                p.GEOM_BOX, halfExtents=[22., 0.55, 0.06],
                rgbaColor=[COULEURS_BANDES[0], COULEURS_BANDES[1], COULEURS_BANDES[2], 1],
                physicsClientId=cid
            )
            p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1,
                              baseVisualShapeIndex=vs_b,
                              basePosition=[0., ry, 0.01], physicsClientId=cid)

        # Bandes d'herbe entre rangées et aux bords
        herbe_y = [-7., -3., 0., 3., 7.]
        for hy in herbe_y:
            vs_h = p.createVisualShape(
                p.GEOM_BOX, halfExtents=[22., 0.9, 0.03],
                rgbaColor=[0.22, 0.55, 0.15, 1],   # vert herbe
                physicsClientId=cid
            )
            p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1,
                              baseVisualShapeIndex=vs_h,
                              basePosition=[0., hy, 0.02], physicsClientId=cid)

        # Chemin de service (piste jaune-beige) sur les bords
        for bord_x in [X_DEBUT - 1.5, X_FIN + 1.5]:
            vs_p = p.createVisualShape(
                p.GEOM_BOX, halfExtents=[0.8, 9., 0.04],
                rgbaColor=[0.75, 0.68, 0.40, 1],
                physicsClientId=cid
            )
            p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1,
                              baseVisualShapeIndex=vs_p,
                              basePosition=[bord_x, 0., 0.02], physicsClientId=cid)

    def _generer_champ(self, cid):
        """Plantes 3D réalistes : tige + feuilles + fleur/épi selon type."""
        import math
        for row_i, pos_y in enumerate(RANGEES_Y):
            for pi in range(30):
                px_ = X_DEBUT + pi * 1.4
                self._creer_plante(cid, px_, pos_y, pi, row_i)

    def _creer_plante(self, cid, x, y, idx, row_i):
        """Une plante composée de plusieurs formes empilées."""
        import math

        # Variation légère de position pour naturel
        ox = (idx * 0.07) % 0.25 - 0.12
        oy = (row_i * 0.11 + idx * 0.05) % 0.20 - 0.10

        # Couleur selon type : maïs (vert foncé) ou légume (vert clair)
        if row_i % 2 == 0:
            col_tige  = [0.15, 0.55, 0.10, 1]   # vert maïs
            col_feuil = [0.20, 0.72, 0.15, 1]
            col_sommet= [0.85, 0.75, 0.10, 1]   # épi doré
            h_tige = 0.65
        else:
            col_tige  = [0.10, 0.48, 0.08, 1]   # vert légume
            col_feuil = [0.18, 0.65, 0.22, 1]
            col_sommet= [0.80, 0.15, 0.10, 1]   # fleur rouge
            h_tige = 0.50

        # ── Tige principale ──────────────────────────────────────────────
        vs_tige = p.createVisualShape(
            p.GEOM_CYLINDER, radius=0.05, length=h_tige,
            rgbaColor=col_tige, physicsClientId=cid
        )
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1,
                          baseVisualShapeIndex=vs_tige,
                          basePosition=[x + ox, y + oy, h_tige / 2],
                          physicsClientId=cid)

        # ── Feuilles (2 ellipsoïdes aplatis orientés en croix) ───────────
        for angle_deg in [0, 90]:
            rad = math.radians(angle_deg)
            vs_f = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[0.30 * abs(math.cos(rad)) + 0.04,
                             0.30 * abs(math.sin(rad)) + 0.04,
                             0.03],
                rgbaColor=col_feuil, physicsClientId=cid
            )
            p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1,
                              baseVisualShapeIndex=vs_f,
                              basePosition=[x + ox, y + oy, h_tige * 0.6],
                              physicsClientId=cid)

        # ── Sommet : épi ou fleur ────────────────────────────────────────
        vs_top = p.createVisualShape(
            p.GEOM_SPHERE, radius=0.10,
            rgbaColor=col_sommet, physicsClientId=cid
        )
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1,
                          baseVisualShapeIndex=vs_top,
                          basePosition=[x + ox, y + oy, h_tige + 0.10],
                          physicsClientId=cid)

    def _creer_drone_simple(self, cid, pos):
        """Drone hexagonal stylisé avec 6 rotors colorés + corps central."""
        import math

        # ── Corps central hexagonal (cylindre aplati) ────────────────────
        vs_corps = p.createVisualShape(
            p.GEOM_CYLINDER, radius=0.50, length=0.12,
            rgbaColor=[0.12, 0.12, 0.12, 1],   # noir carbone
            physicsClientId=cid
        )
        cs_corps = p.createCollisionShape(
            p.GEOM_CYLINDER, radius=0.50, height=0.12,
            physicsClientId=cid
        )

        # ── Dôme supérieur (électronique) ─────────────────────────────────
        vs_dome = p.createVisualShape(
            p.GEOM_SPHERE, radius=0.22,
            rgbaColor=[0.18, 0.45, 0.85, 1],   # bleu électronique
            physicsClientId=cid
        )

        # ── Réservoir de pulvérisation (sous le corps) ────────────────────
        vs_reservoir = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[0.18, 0.10, 0.14],
            rgbaColor=[0.85, 0.92, 0.98, 1],   # blanc laiteux
            physicsClientId=cid
        )

        drone_id = p.createMultiBody(
            baseMass=MASSE_INITIALE,
            baseCollisionShapeIndex=cs_corps,
            baseVisualShapeIndex=vs_corps,
            basePosition=pos,
            physicsClientId=cid
        )

        # Dôme et réservoir en objets séparés (visuels seulement, mass=0)
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1,
                          baseVisualShapeIndex=vs_dome,
                          basePosition=[pos[0], pos[1], pos[2] + 0.16],
                          physicsClientId=cid)
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1,
                          baseVisualShapeIndex=vs_reservoir,
                          basePosition=[pos[0], pos[1], pos[2] - 0.16],
                          physicsClientId=cid)

        # ── 6 Bras + Rotors ───────────────────────────────────────────────
        BRAS_LONGUEUR = 0.85
        COULEURS_ROTORS = [
            [1.0, 0.25, 0.10, 1],  # rouge
            [1.0, 0.25, 0.10, 1],
            [0.15, 0.85, 0.25, 1], # vert
            [0.15, 0.85, 0.25, 1],
            [0.95, 0.85, 0.10, 1], # jaune
            [0.95, 0.85, 0.10, 1], # (paires de couleurs = paires de moteurs)
        ]

        for i in range(6):
            angle = math.radians(i * 60)
            rx = pos[0] + BRAS_LONGUEUR * math.cos(angle)
            ry = pos[1] + BRAS_LONGUEUR * math.sin(angle)
            rz = pos[2] + 0.02

            # Bras
            vs_bras = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[BRAS_LONGUEUR / 2, 0.04, 0.03],
                rgbaColor=[0.20, 0.20, 0.20, 1],
                physicsClientId=cid
            )
            bx = pos[0] + (BRAS_LONGUEUR / 2) * math.cos(angle)
            by = pos[1] + (BRAS_LONGUEUR / 2) * math.sin(angle)
            orn_bras = p.getQuaternionFromEuler([0, 0, angle])
            p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1,
                              baseVisualShapeIndex=vs_bras,
                              basePosition=[bx, by, pos[2]],
                              baseOrientation=orn_bras,
                              physicsClientId=cid)

            # Nacelle moteur
            vs_nacelle = p.createVisualShape(
                p.GEOM_CYLINDER, radius=0.10, length=0.08,
                rgbaColor=[0.30, 0.30, 0.30, 1],
                physicsClientId=cid
            )
            p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1,
                              baseVisualShapeIndex=vs_nacelle,
                              basePosition=[rx, ry, rz],
                              physicsClientId=cid)

            # Rotor (disque translucide coloré)
            vs_rotor = p.createVisualShape(
                p.GEOM_CYLINDER, radius=0.32, length=0.02,
                rgbaColor=COULEURS_ROTORS[i],
                physicsClientId=cid
            )
            p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1,
                              baseVisualShapeIndex=vs_rotor,
                              basePosition=[rx, ry, rz + 0.06],
                              physicsClientId=cid)

        # ── LED de statut (sphère verte sur le nez) ───────────────────────
        vs_led = p.createVisualShape(
            p.GEOM_SPHERE, radius=0.06,
            rgbaColor=[0.10, 1.0, 0.30, 1],
            physicsClientId=cid
        )
        nez_x = pos[0] + 0.52
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1,
                          baseVisualShapeIndex=vs_led,
                          basePosition=[nez_x, pos[1], pos[2] + 0.06],
                          physicsClientId=cid)

        return drone_id

    def render(self):
        pass

    def close(self):
        if self._physics_client is not None:
            try:
                p.disconnect(self._physics_client)
            except Exception:
                pass
            self._physics_client = None


# Enregistrement Gym
try:
    from gymnasium.envs.registration import register
    register(id="DroneAgricole-v1",
             entry_point="drone_env:DroneAgricoleEnv",
             max_episode_steps=MAX_STEPS)
except Exception:
    pass
