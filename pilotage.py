import pybullet as p
import pybullet_data
import time
import math
import csv
import random
import os  

# --- 0. CONFIGURATION DES DOSSIERS DE SORTIE ---
# Définition du chemin vers le dossier data
output_dir = "data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Dossier '{output_dir}' créé à la racine.")

# Chemins complets pour les fichiers
csv_path = os.path.join(output_dir, "rapport_mission_agricole.csv")
log_path = os.path.join(output_dir, "log_simulation.txt")

# Ouverture du fichier log pour enregistrer les événements
log_file = open(log_path, "w")

def log_event(message):
    timestamp = time.strftime("%H:%M:%S")
    formatted_msg = f"[{timestamp}] {message}"
    print(formatted_msg)
    log_file.write(formatted_msg + "\n")

# --- 1. INITIALISATION DE LA SIMULATION ---
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.loadURDF("plane.urdf")

# Chargement du drone
droneId = p.loadURDF("models/agri_hexacopter.urdf", [0, 0, 0.4])

# Libération des moteurs
for i in range(p.getNumJoints(droneId)):
    p.setJointMotorControl2(droneId, i, p.VELOCITY_CONTROL, force=0)

# --- 2. PARAMÈTRES DE VOL ET PHYSIQUE ---
MASSE_INITIALE = 15.0
GRAVITE = 9.81
DT = 1./240.
kp_z, kd_z = 100.0, 60.0
kp_att, kd_att = 45.0, 35.0
ALTITUDE_TRAVAIL = 15.0
ANGLE_MANOEUVRE = 0.20
debit_pulverisation = 0.05

# --- 3. PRÉPARATION DU RECUEIL DE DONNÉES (Dans le dossier data) ---
file = open(csv_path, mode='w', newline='')
writer = csv.writer(file)
writer.writerow(['Temps', 'Altitude_Radar_m', 'Niveau_Reservoir_L', 'Indice_NDVI', 'Masse_Actuelle_kg'])

# --- 4. BOUCLE PRINCIPALE ET MACHINE À ÉTATS ---
state = "WARMUP"
masse_actuelle = MASSE_INITIALE
start_time = time.time()
p.setTimeStep(DT)

try:
    log_event(f"Système Agri-Drone prêt. État : {state}")
    while True:
        pos, orn = p.getBasePositionAndOrientation(droneId)
        lin_vel, ang_vel = p.getBaseVelocity(droneId)
        euler = p.getEulerFromQuaternion(orn)
        now = time.time() - start_time

        target_z = 0.0
        target_roll = 0.0
        pulverisation_active = False

        if state == "WARMUP":
            if now > 3.0: 
                state = "TAKEOFF"
                log_event("Changement d'état : TAKEOFF")

        elif state == "TAKEOFF":
            target_z = ALTITUDE_TRAVAIL
            if pos[2] > (ALTITUDE_TRAVAIL - 0.5):
                state = "SPRAYING_RIGHT"
                start_time = time.time()
                log_event("Début pulvérisation : Direction Droite")

        elif state == "SPRAYING_RIGHT":
            target_z = ALTITUDE_TRAVAIL
            target_roll = -ANGLE_MANOEUVRE
            pulverisation_active = True
            if now > 15.0:
                state = "SPRAYING_LEFT"
                start_time = time.time()
                log_event("Changement d'état : SPRAYING_LEFT (Direction Gauche)")

        elif state == "SPRAYING_LEFT":
            target_z = ALTITUDE_TRAVAIL
            target_roll = ANGLE_MANOEUVRE
            pulverisation_active = True
            if now > 30.0:
                state = "LANDING"
                log_event("Mission terminée, retour au sol.")

        elif state == "LANDING":
            target_z = 0.35
            pulverisation_active = False
            if pos[2] < 0.45:
                log_event("Drone posé au sol.")
                break

        # C. Simulation de la consommation
        if pulverisation_active and masse_actuelle > 10.0:
            masse_actuelle -= debit_pulverisation * DT
            p.changeDynamics(droneId, -1, mass=masse_actuelle)

        # D. Contrôleur PID (Adaptatif)
        if state != "WARMUP":
            poussee_necessaire = masse_actuelle * GRAVITE
            erreur_z = target_z - pos[2]
            force_z = (erreur_z * kp_z) - (lin_vel[2] * kd_z) + poussee_necessaire
        else:
            force_z = (masse_actuelle * GRAVITE) * 0.6

        # E. Stabilisation d'Assiette
        corr_roll  = (target_roll - euler[0]) * kp_att - (ang_vel[0] * kd_att)
        corr_pitch = (0 - euler[1]) * kp_att - (ang_vel[1] * kd_att)
        corr_yaw   = (0 - euler[2]) * kp_att - (ang_vel[2] * kd_att)

        p.applyExternalForce(droneId, -1, [0, 0, force_z], [0, 0, 0], p.LINK_FRAME)
        p.applyExternalTorque(droneId, -1, [corr_roll, corr_pitch, corr_yaw], p.LINK_FRAME)

        # G. Recueil des données Agricoles
        dist_radar = pos[2] - 1.0 
        ndvi = 0.8 if (pos[0] < 2) else 0.3 + random.uniform(-0.05, 0.05)
        niveau_tank = ((masse_actuelle - 10.0) / 5.0) * 100

        writer.writerow([round(now, 2), round(dist_radar, 2), round(niveau_tank, 1), round(ndvi, 2), round(masse_actuelle, 2)])

        # H. Animation des hélices
        for i in range(p.getNumJoints(droneId)):
            joint_name = p.getJointInfo(droneId, i)[1]
            if b"j_p" in joint_name: 
                p.setJointMotorControl2(droneId, i, p.VELOCITY_CONTROL, targetVelocity=150)

        p.stepSimulation()
        time.sleep(DT)

except KeyboardInterrupt:
    log_event("Simulation interrompue par l'utilisateur.")

finally:
    file.close()
    log_file.close()
    p.disconnect()
    print(f"\n✅ Mission terminée.")
    print(f"📊 CSV stocké : {csv_path}")
    print(f"📝 Log stocké : {log_path}")