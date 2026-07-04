import pybullet as p
import pybullet_data
import time
import math
import random
import csv

# --- 1. INITIALISATION DE LA SIMULATION ---
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.loadURDF("plane.urdf")

# --- 2. GÉNÉRATION DES RANGÉES DE CULTURES ---
# Création de 4 rangées distinctes et alignées
ESPACEMENT_RANGEES = 3.0

def generer_champ_rangees():
    largeur_plant = 0.2
    hauteur_plant = 1.0
    culture_ids = []
    
    # Coordonnées Y des 4 rangées successives
    positions_y = [-4.5, -1.5, 1.5, 4.5]
    
    for r_idx, pos_y in enumerate(positions_y):
        # Chaque rangée fait 40 mètres de long (de X = -20m à X = 20m)
        for p_idx in range(30):
            pos_x = -20.0 + (p_idx * 1.4)
            
            # Alternance de couleurs (Vert = sain, Jaune/Brun = besoin d'engrais)
            couleur = [0.1, 0.7, 0.2, 1] if (p_idx % 5 != 0) else [0.6, 0.5, 0.2, 1]
            
            visual_shape = p.createVisualShape(
                shapeType=p.GEOM_BOX, 
                halfExtents=[largeur_plant, largeur_plant, hauteur_plant / 2], 
                rgbaColor=couleur
            )
            collision_shape = p.createCollisionShape(
                shapeType=p.GEOM_BOX, 
                halfExtents=[largeur_plant, largeur_plant, hauteur_plant / 2]
            )
            
            plant_id = p.createMultiBody(
                baseMass=0, 
                baseCollisionShapeIndex=collision_shape, 
                baseVisualShapeIndex=visual_shape, 
                basePosition=[pos_x, pos_y, hauteur_plant / 2]
            )
            culture_ids.append(plant_id)
    return culture_ids

champ_plants = generer_champ_rangees()

# --- 3. CHARGEMENT DU DRONE ---
# Le drone commence parfaitement aligné sur la Rangée 1 (X = -22m, Y = -4.5m)
droneId = p.loadURDF("models/agri_hexacopter.urdf", [-22.0, -4.5, 0.4])

# Libération des liaisons moteurs pour un pilotage direct par forces externes
for i in range(p.getNumJoints(droneId)):
    p.setJointMotorControl2(droneId, i, p.VELOCITY_CONTROL, force=0)

# --- 4. PARAMÈTRES PHYSIQUES ET ENREGISTREMENT ---
MASSE_INITIALE = 15.0
GRAVITE = 9.81
DT = 1./240.
p.setTimeStep(DT)

# Coefficients du contrôleur PID de stabilité
kp_z, kd_z = 100.0, 60.0
kp_att, kd_att = 45.0, 35.0

ALTITUDE_TRAVAIL = 15.0
VITESSE_INCLINAISON = 0.12  # Angle pour déplacer le drone de 15 kg avec inertie
debit_pulverisation = 0.03  # Consommation lissée pour durer 2 minutes

# Préparation du fichier CSV pour recueillir les données de position et de réservoir
file = open('mission_agricole_rangees.csv', mode='w', newline='')
writer = csv.writer(file)
writer.writerow(['Temps', 'Altitude_m', 'Position_X', 'Position_Y', 'Masse_kg', 'Etat_Mission'])

# --- 5. INITIALISATION DU PLANIFICATEUR DE MISSION ---
state = "WARMUP"
masse_actuelle = MASSE_INITIALE
start_time = time.time()

# Placement optimal de la caméra de simulation pour observer les lignes d'arrosage
p.resetDebugVisualizerCamera(cameraDistance=28, cameraYaw=40, cameraPitch=-35, cameraTargetPosition=[0, 0, 5])

try:
    print("Plan de vol activé : Traitement ordonné par rangées successives (Cible : 2 min).")
    while True:
        # A. Lecture en temps réel des capteurs d'état (IMU / Altimètre / GPS virtuel)
        pos, orn = p.getBasePositionAndOrientation(droneId)
        lin_vel, ang_vel = p.getBaseVelocity(droneId)
        euler = p.getEulerFromQuaternion(orn)
        now = time.time() - start_time

        target_z = ALTITUDE_TRAVAIL
        target_roll = 0.0
        target_pitch = 0.0
        pulverisation_active = False

        # --- B. SÉQUENCEUR ET MACHINE À ÉTATS DE LA MISSION ---
        if state == "WARMUP":
            target_z = 0.0
            if now > 3.0: 
                state = "TAKEOFF"
                print(f"[{round(now,1)}s] Phase 1 : Décollage vertical vers {ALTITUDE_TRAVAIL}m.")

        elif state == "TAKEOFF":
            target_z = ALTITUDE_TRAVAIL
            if pos[2] > (ALTITUDE_TRAVAIL - 0.5):
                state = "ROW1_FORWARD"
                print(f"[{round(now,1)}s] Phase 2 : Début RANGÉE 1 (Y = -4.5m) -> Traitement en avant.")

        # ---- RANGÉE 1 ----
        elif state == "ROW1_FORWARD":
            target_pitch = VITESSE_INCLINAISON
            pulverisation_active = True
            if pos[0] > 20.0:  # Arrivé au bout de la première ligne
                state = "SHIFT_TO_ROW2"
                print(f"[{round(now,1)}s] Fin Rangée 1. Coupure arrosage et décalage vers Rangée 2.")

        # ---- DÉCALAGE VERS LIGNE 2 ----
        elif state == "SHIFT_TO_ROW2":
            target_roll = VITESSE_INCLINAISON  # Glisse sur la gauche (Axe Y positif)
            if pos[1] >= -1.5:  # Aligné sur l'axe Y de la rangée 2
                state = "ROW2_BACKWARD"
                print(f"[{round(now,1)}s] Phase 3 : Début RANGÉE 2 (Y = -1.5m) -> Traitement en arrière.")

        # ---- RANGÉE 2 ----
        elif state == "ROW2_BACKWARD":
            target_pitch = -VITESSE_INCLINAISON
            pulverisation_active = True
            if pos[0] < -20.0:
                state = "SHIFT_TO_ROW3"
                print(f"[{round(now,1)}s] Fin Rangée 2. Coupure arrosage et décalage vers Rangée 3.")

        # ---- DÉCALAGE VERS LIGNE 3 ----
        elif state == "SHIFT_TO_ROW3":
            target_roll = VITESSE_INCLINAISON
            if pos[1] >= 1.5:
                state = "ROW3_FORWARD"
                print(f"[{round(now,1)}s] Phase 4 : Début RANGÉE 3 (Y = 1.5m) -> Traitement en avant.")

        # ---- RANGÉE 3 ----
        elif state == "ROW3_FORWARD":
            target_pitch = VITESSE_INCLINAISON
            pulverisation_active = True
            if pos[0] > 20.0:
                state = "SHIFT_TO_ROW4"
                print(f"[{round(now,1)}s] Fin Rangée 3. Coupure arrosage et décalage vers la dernière Rangée.")

        # ---- DÉCALAGE VERS LIGNE 4 ----
        elif state == "SHIFT_TO_ROW4":
            target_roll = VITESSE_INCLINAISON
            if pos[1] >= 4.5:
                state = "ROW4_BACKWARD"
                print(f"[{round(now,1)}s] Phase 5 : Début RANGÉE 4 (Y = 4.5m) -> Traitement de clôture.")

        # ---- RANGÉE 4 ----
        elif state == "ROW4_BACKWARD":
            target_pitch = -VITESSE_INCLINAISON
            pulverisation_active = True
            # Sécurité temporelle : si on dépasse 112 secondes, on rentre à la base
            if pos[0] < -20.0 or now > 112.0:
                state = "LANDING"
                print(f"[{round(now,1)}s] Fin globale du traitement agricole. Retour au sol en cours.")

        # ---- ATTERRISSAGE DURÉE TOTAL ~2 MIN ----
        elif state == "LANDING":
            target_z = 0.35  # Descente vers la hauteur des pieds
            if pos[2] < 0.45 or now > 120.0:
                print(f"[{round(now,1)}s] Mission accomplie avec succès. Drone immobilisé.")
                break

        # --- C. PHYSIQUE ET CONTRÔLE DE VOL ---
        # Simulation en temps réel du vidage du réservoir (perte de masse)
        if pulverisation_active and masse_actuelle > 10.0:
            masse_actuelle -= debit_pulverisation * DT
            p.changeDynamics(droneId, -1, mass=masse_actuelle)

        # Calcul de la poussée verticale PID (s'adapte à la masse changeante)
        if state != "WARMUP":
            poussee_g = masse_actuelle * GRAVITE
            force_z = ((target_z - pos[2]) * kp_z) - (lin_vel[2] * kd_z) + poussee_g
        else:
            force_z = (masse_actuelle * GRAVITE) * 0.5

        # Algorithme d'Auto-Leveling d'assiette (Maintien de l'orientation cible)
        corr_roll  = (target_roll - euler[0]) * kp_att - (ang_vel[0] * kd_att)
        corr_pitch = (target_pitch - euler[1]) * kp_att - (ang_vel[1] * kd_att)
        corr_yaw   = (0 - euler[2]) * kp_att - (ang_vel[2] * kd_att)

        # Application des forces de propulsion et des couples stabilisateurs
        p.applyExternalForce(droneId, -1, [0, 0, force_z], [0, 0, 0], p.LINK_FRAME)
        p.applyExternalTorque(droneId, -1, [corr_roll, corr_pitch, corr_yaw], p.LINK_FRAME)

        # --- D. COLLECTE ET SAUVEGARDE DES DONNÉES DES CAPTEURS ---
        writer.writerow([
            round(now, 2), 
            round(pos[2], 2), 
            round(pos[0], 2), 
            round(pos[1], 2), 
            round(masse_actuelle, 2), 
            state
        ])

        # Animation visuelle synchrone des hélices de l'hexacoptère
        for i in range(p.getNumJoints(droneId)):
            if b"j_p" in p.getJointInfo(droneId, i)[1]:
                p.setJointMotorControl2(droneId, i, p.VELOCITY_CONTROL, targetVelocity=140)

        p.stepSimulation()
        time.sleep(DT)

except KeyboardInterrupt:
    pass
finally:
    # Fermeture propre et enregistrement définitif du fichier d'analyse
    file.close()
    p.disconnect()
    print("Fichier de résultats agricoles sauvegardé sous 'mission_agricole_rangees.csv'.")