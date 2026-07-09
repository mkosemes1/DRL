"""
  DRONE_ENV.PY  —  Environnement Gymnasium pour Drone Agricole Hexacoptère
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

# ─────────────────────── Santé des plantes & pulvérisation ciblée ─────────────
# La v4 liait la récompense de pulvérisation UNIQUEMENT à la masse dépensée,
# sans vérifier qu'il y avait réellement une plante sous le drone ni qu'elle en
# avait besoin — un agent aurait pu "pulvériser" au-dessus d'un sol déjà traité
# et être récompensé de la même façon. Cette v5 simule un état de santé réel
# par plante (analogue simplifié d'un indice de vigueur type NDVI, dans [0,1])
# et ne récompense QUE l'amélioration effective de cet état.
NB_PLANTES_PAR_RANGEE = 30
ESPACEMENT_PLANTES    = 1.4
PLANTES_X = X_DEBUT + np.arange(NB_PLANTES_PAR_RANGEE) * ESPACEMENT_PLANTES  # positions X fixes (30,)

SANTE_INIT_MIN   = 0.25   # santé minimale au reset (état de stress du champ observé au début de la mission)
SANTE_INIT_MAX   = 0.65   # santé maximale au reset — jamais 1.0 : il y a toujours un vrai travail à faire
SPRAY_RADIUS_X   = 2.5    # m — rayon effectif du cône de pulvérisation le long de X
TAUX_TRAITEMENT  = 2.0    # gain de santé/s au centre du cône (décroît linéairement jusqu'au bord)
# Calibrage vérifié numériquement (script de simulation à vitesse de croisière
# ~13 m/s) : un unique passage rectiligne porte la santé moyenne d'une rangée
# de ~0.45 à ~0.82 ; un passage deux fois plus lent la porte à ~0.98 — un vrai
# compromis vitesse/qualité de traitement, pas un simple "case à cocher".

# ─────────────────────── Paramètres RL ───────────────────────────────────────
MAX_STEPS       = 4000
TOLÉRANCE_Y     = 0.5
TOLÉRANCE_Z     = 1.0
SEUIL_CRASH_Z   = 0.25

# ─────────────────────── Récompenses (v4 — PBRS) ─────────────────────────────
# Deux familles de termes, volontairement séparées et d'échelle comparable
# (cf. manuel §3.9.4 "Common pitfalls in multi-reward composition" — le piège
# n°1 est le "scale mismatch" : un terme trop grand écrase tous les autres) :
#
#   1) Objectifs RÉELS de la tâche -> récompenses éparses (sparse), déclenchées
#      seulement par un événement (fin de rangée / mission / crash).
#   2) Signal d'apprentissage dense -> UN SEUL terme de shaping potential-based
#      (PBRS), dont la somme actualisée sur l'épisode est mathématiquement bornée.
#
# GAMMA : le théorème de Ng, Harada & Russell (1999) exige F(s,a,s') = γΦ(s')−Φ(s)
# avec le MÊME γ que celui utilisé par l'agent pour actualiser ses retours.
# GAMMA est donc défini ICI comme source unique de vérité, et train_drl.py
# l'importe pour configurer PPO — évite que l'environnement et l'algorithme
# divergent silencieusement, ce qui invaliderait la garantie d'invariance.
#
# Valeur choisie (0.9995, pas 0.99 ni 0.995) : l'horizon effectif d'un agent
# actualisé est ~1/(1-γ). Avec MAX_STEPS=4000 et des récompenses de tâche
# concentrées en fin d'épisode (fin de rangée, mission), un γ trop faible
# les efface presque totalement du retour actualisé vu depuis le début de
# l'épisode. Vérifié numériquement pendant le développement : avec γ=0.99
# OU γ=0.995, le retour actualisé d'une mission réussie devient INFÉRIEUR à
# celui d'un drone qui reste immobile — la garantie d'invariance de politique
# du PBRS (V'^π(s) = V^π(s) − Φ(s), un décalage constant qui ne doit jamais
# changer l'ordre des politiques) tient toujours algébriquement, mais devient
# numériquement inutile si l'horizon effectif de γ est plus court que
# l'horizon réel de la tâche. γ=0.9995 (horizon effectif ≈ 2000 pas) restaure
# une marge saine sur toute la plage réaliste de durées de mission.
GAMMA           =   0.9995  # doit être identique au "gamma" passé à PPO
K_SHAPING       =   5.0    # gain du terme de shaping potential-based (dense)
R_FIN_RANGEE    =  20.0    # objectif intermédiaire réel (sparse)
R_MISSION_FINIE = 100.0    # objectif final réel (sparse) — même ordre que P_CRASH
R_SANTE_PAR_UNITE = 1.0    # récompense ∝ santé RÉELLEMENT restaurée (sparse,
                            # bornée par le déficit total du champ au reset)
P_CRASH         = -100.0   # pénalité terminale, calibrée à l'échelle de R_MISSION_FINIE
P_DIVERGENCE    = -20.0    # sortie de la zone de vol autorisée
P_INCLINAISON   =  -0.05   # coût de stabilité par pas (faible poids, cf. BipedalWalker)
P_CONTROLE      =  -0.001  # coût de contrôle style HalfCheetah (∝ ||action||²),
                            # décourage les commandes brusques/énergivores
P_TEMPS         =  -0.002  # petit coût par pas ("Rule B" du manuel) : à récompense
                            # de tâche égale, un épisode plus court est préféré


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
            MASSE_MIN, 0., -45., -10., -20., 0.,
            0., 0., 0.,                                 # sante_locale, sante_rangee, sante_globale
        ], dtype=np.float32)
        obs_high = np.array([
             30., 10., 30.,
             np.pi, np.pi, np.pi,
             10., 10., 10.,
             5., 5., 5.,
             MASSE_INITIALE, 3., 45., 10., 20., 1.,
             1., 1., 1.,
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
        self._phi_prev       = 0.0
        self._stats         = {}

    # ── Fonction de potentiel Φ(s) — reward shaping potential-based (PBRS) ──────
    def _potentiel(self, x, y, z, target_x, target_y):
        """Φ(s) : plus proche de 0 = plus proche de l'objectif courant.

        Combine 3 distances normalisées en [0, 1] (avancement le long de la
        rangée, alignement latéral, écart à l'altitude de travail). Ne dépend
        QUE de l'état courant (pas de l'action, pas de l'historique), ce qui
        est la condition requise par Ng, Harada & Russell (1999) pour que
        F(s,s') = γΦ(s') − Φ(s) ne modifie pas la politique optimale.
        """
        d_x = abs(target_x - x) / (X_FIN - X_DEBUT)
        d_y = abs(y - target_y) / 10.0
        d_z = abs(z - ALTITUDE_TRAVAIL) / ALTITUDE_TRAVAIL
        return -(d_x + 0.6 * d_y + 0.6 * d_z)

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

        # État de santé initial du champ (analogue simplifié d'un indice de
        # vigueur type NDVI, dans [0,1]) : généré AVANT le champ pour que les
        # plantes soient coloriées dès la création selon leur état de stress.
        self._sante = self.np_random.uniform(
            SANTE_INIT_MIN, SANTE_INIT_MAX,
            size=(len(RANGEES_Y), NB_PLANTES_PAR_RANGEE)
        ).astype(np.float32)

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
        self._stats = {"total_pulverise": 0., "rangees_finies": 0,
                       "crash": False, "steps": 0,
                       "sante_globale_initiale": float(self._sante.mean()),
                       "sante_totale_restauree": 0.}

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

        # Point de départ de la fonction de potentiel (état réel post-préchauffage,
        # pas la position théorique de spawn) pour un télescopage exact du shaping.
        pos0, _ = p.getBasePositionAndOrientation(self._drone_id, physicsClientId=cid)
        target_x0 = X_FIN if self._dir_forward else X_DEBUT
        target_y0 = RANGEES_Y[self._rangee_idx]
        self._phi_prev = self._potentiel(pos0[0], pos0[1], pos0[2], target_x0, target_y0)

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

        # Pulvérisation ciblée — récompense proportionnelle à la SANTÉ
        # RÉELLEMENT RESTAURÉE sur les plantes effectivement couvertes par le
        # cône de pulvérisation, pas simplement au volume de produit dépensé.
        # Différence avec la v4 : avant, "pulvériser au bon endroit
        # géométrique" suffisait à être récompensé, même si le sol y était
        # déjà traité ou si aucune plante stressée ne s'y trouvait — un agent
        # aurait pu apprendre à "faire semblant" de traiter. Ici, la
        # récompense ne peut venir que d'une amélioration MESURÉE de l'état
        # ([0,1], analogue simplifié d'un indice de vigueur type NDVI) des
        # plantes réellement survolées. Ce terme reste intrinsèquement borné :
        # chaque plante plafonne à santé=1.0 et ne peut plus rapporter de
        # récompense une fois soignée, donc la somme sur tout l'épisode ne
        # peut jamais dépasser R_SANTE_PAR_UNITE × (déficit total du champ au
        # reset), quelle que soit la durée de l'épisode.
        self._pulv_active = (abs(z - ALTITUDE_TRAVAIL) < TOLÉRANCE_Z and
                             abs(y - target_y) < TOLÉRANCE_Y and z > 2.0)
        r_sante = 0.0
        if self._pulv_active and self._masse > MASSE_MIN:
            kg_pulverise = DEBIT_PULV * DT
            self._masse -= kg_pulverise
            self._stats["total_pulverise"] += kg_pulverise
            p.changeDynamics(self._drone_id, -1, mass=self._masse, physicsClientId=cid)

            row = self._rangee_idx
            dists = np.abs(PLANTES_X - x)
            mask = dists <= SPRAY_RADIUS_X
            if np.any(mask):
                efficacite = np.clip(1.0 - dists[mask] / SPRAY_RADIUS_X, 0.0, 1.0)
                dose   = TAUX_TRAITEMENT * DT * efficacite
                avant  = self._sante[row, mask].copy()
                self._sante[row, mask] = np.minimum(1.0, avant + dose)
                delta  = self._sante[row, mask] - avant           # gain réel, 0 si déjà à 1.0
                r_sante = R_SANTE_PAR_UNITE * float(np.sum(delta))
                self._stats["sante_totale_restauree"] += float(np.sum(delta))
                if self.render_mode == "human":
                    for pi in np.nonzero(mask)[0]:
                        self._maj_couleur_plante(row, int(pi))

        # Fin de rangée (objectif réel intermédiaire -> sparse)
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
        mission_finie  = self._rangees_faites >= 4

        # ═══════════════════════════════════════════════════════════════════
        # RÉCOMPENSE — deux familles cohérentes en échelle (voir README) :
        #   (a) shaping dense potential-based (PBRS), borné par télescopage
        #   (b) récompenses/pénalités éparses sur les vrais objectifs de la tâche
        # ═══════════════════════════════════════════════════════════════════

        # (a) Shaping PBRS strict : F(s,a,s') = γ·Φ(s') − Φ(s) (Ng, Harada &
        # Russell, 1999), avec le MÊME γ que PPO (voir la constante GAMMA
        # ci-dessus, importée par train_drl.py). C'est cette cohérence γ qui
        # garantit V'^π(s) = V^π(s) − Φ(s) pour toute politique π : un décalage
        # CONSTANT (à s fixé) qui ne dépend pas de π, donc qui ne change jamais
        # l'ordre des politiques ni la politique optimale — seulement l'échelle
        # apparente des retours. La cible (target_x/target_y) peut changer au
        # pas où une rangée se termine ; c'est un saut de sous-objectif
        # volontaire (curriculum de cibles), pas une fuite de récompense — la
        # rangée terminée est de toute façon déjà récompensée explicitement
        # par R_FIN_RANGEE ci-dessous.
        phi_now  = self._potentiel(x, y, z, target_x, target_y)
        r_shaping = K_SHAPING * (GAMMA * phi_now - self._phi_prev)
        self._phi_prev = phi_now

        # (b) Coûts de contrôle et de temps (style HalfCheetah / "Rule B")
        r_controle = P_CONTROLE * float(np.sum(np.square(action)))
        r_temps    = P_TEMPS
        incli      = abs(euler[0]) + abs(euler[1])
        r_incli    = P_INCLINAISON * incli

        reward = r_shaping + r_sante + r_controle + r_temps + r_incli

        if rangee_finie:
            reward += R_FIN_RANGEE
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

        # Signaux de santé du champ — indispensables pour que l'agent puisse
        # PERCEVOIR où traiter en priorité (sinon la récompense de santé
        # serait un signal sans levier observable, inutilisable par PPO).
        row = self._rangee_idx
        dists = np.abs(PLANTES_X - x)
        mask_locale = dists <= SPRAY_RADIUS_X
        sante_locale  = float(self._sante[row, mask_locale].mean()) if np.any(mask_locale) \
                        else float(self._sante[row].mean())
        sante_rangee  = float(self._sante[row].mean())
        sante_globale = float(self._sante.mean())

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
            sante_locale, sante_rangee, sante_globale,
        ], dtype=np.float32)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def _get_info(self):
        return {
            "rangee": self._rangee_idx,
            "rangees_finies": self._rangees_faites,
            "masse": round(self._masse, 3),
            "pulv_active": self._pulv_active,
            "step": self._step_count,
            "sante_globale": round(float(self._sante.mean()), 4),
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
        """Plantes 3D réalistes : tige + feuilles + fleur/épi selon type.
        Les feuilles sont coloriées selon self._sante et leurs IDs conservés
        dans self._plantes_ids pour pouvoir les recolorier pendant le vol
        (visualisation uniquement — voir _maj_couleur_plante)."""
        self._plantes_ids = [[None] * NB_PLANTES_PAR_RANGEE for _ in RANGEES_Y]
        for row_i, pos_y in enumerate(RANGEES_Y):
            for pi in range(NB_PLANTES_PAR_RANGEE):
                px_ = PLANTES_X[pi]
                self._plantes_ids[row_i][pi] = self._creer_plante(
                    cid, px_, pos_y, pi, row_i, float(self._sante[row_i, pi])
                )

    def _couleur_sante(self, niveau, col_sain):
        """Interpole une couleur RGBA entre 'stressé' (jaune-brun, niveau=0)
        et col_sain (niveau=1) — analogue visuel d'un indice NDVI."""
        col_stress = [0.55, 0.42, 0.12, 1.0]
        t = float(np.clip(niveau, 0.0, 1.0))
        return [col_stress[k] * (1 - t) + col_sain[k] * t for k in range(4)]

    def _creer_plante(self, cid, x, y, idx, row_i, sante_val):
        """Une plante composée de plusieurs formes empilées.
        Retourne les IDs des 2 corps 'feuilles' (recoloriés selon la santé)."""
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
        # Couleur initiale = santé au reset (jaune-brun si stressée, verte si saine)
        couleur_feuille = self._couleur_sante(sante_val, col_feuil)
        leaf_ids = []
        for angle_deg in [0, 90]:
            rad = math.radians(angle_deg)
            vs_f = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[0.30 * abs(math.cos(rad)) + 0.04,
                             0.30 * abs(math.sin(rad)) + 0.04,
                             0.03],
                rgbaColor=couleur_feuille, physicsClientId=cid
            )
            leaf_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1,
                              baseVisualShapeIndex=vs_f,
                              basePosition=[x + ox, y + oy, h_tige * 0.6],
                              physicsClientId=cid)
            leaf_ids.append(leaf_id)

        # ── Sommet : épi ou fleur ────────────────────────────────────────
        vs_top = p.createVisualShape(
            p.GEOM_SPHERE, radius=0.10,
            rgbaColor=col_sommet, physicsClientId=cid
        )
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1,
                          baseVisualShapeIndex=vs_top,
                          basePosition=[x + ox, y + oy, h_tige + 0.10],
                          physicsClientId=cid)

        return leaf_ids

    def _maj_couleur_plante(self, row_i, pi):
        """Recolorie les feuilles d'une plante selon sa santé courante.
        Appelé uniquement en render_mode='human' (coût GPU/CPU inutile en
        entraînement headless, où rien n'est jamais affiché)."""
        cid = self._physics_client
        col_sain = [0.20, 0.72, 0.15, 1] if row_i % 2 == 0 else [0.18, 0.65, 0.22, 1]
        couleur = self._couleur_sante(float(self._sante[row_i, pi]), col_sain)
        for leaf_id in self._plantes_ids[row_i][pi]:
            p.changeVisualShape(leaf_id, -1, rgbaColor=couleur, physicsClientId=cid)

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
