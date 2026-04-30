import pybullet as p
import pybullet_data
import time

def run_simulation():
    # 1. Connexion au moteur physique (GUI = Interface Graphique)
    physicsClient = p.connect(p.GUI)
    
    # Ajout d'une option pour que la caméra ne bouge pas toute seule
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
    
    # 2. Ressources par défaut (sol)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # 3. Gravité réelle
    p.setGravity(0, 0, -9.81)
    
    # 4. Chargement du sol
    planeId = p.loadURDF("plane.urdf")
    
    # 5. Chargement de VOTRE drone hexacoptère
    # Nous le plaçons à 1.5m pour bien voir l'impact des pieds "H" au sol
    droneStartPos = [0, 0, 1.5]
    droneStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
    
    try:
        # Assurez-vous que le fichier est bien nommé agri_hexacopter.urdf dans models/
        droneId = p.loadURDF("models/agri_hexacopter.urdf", 
                             droneStartPos, 
                             droneStartOrientation,
                             useFixedBase=False)
        print(f"✅ Drone chargé avec succès (ID: {droneId})")
    except Exception as e:
        print(f"❌ Erreur de chargement : {e}")
        return

    # 6. Boucle de simulation infinie
    print("🚀 Simulation active. Fermez la fenêtre pour arrêter.")
    
    try:
        while p.isConnected(): # Continue tant que la fenêtre est ouverte
            p.stepSimulation()
            time.sleep(1./240.) # Fréquence de 240Hz pour une physique fluide
    except p.error:
        print("Simulation interrompue.")
    finally:
        p.disconnect()

if __name__ == "__main__":
    run_simulation()