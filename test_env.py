"""
=============================================================================
  TEST_ENV.PY  —  Test rapide de la physique avant entraînement
  DIT Dakar — Projet DRL Drone Agricole
=============================================================================
Utilisation :
  python test_env.py            # Test silencieux (sans GUI)
  python test_env.py --gui      # Avec fenêtre PyBullet (voir le drone)
  python test_env.py --gui --steps 500
"""

import argparse
import time
import numpy as np
from drone_env import DroneAgricoleEnv, SEUIL_CRASH_Z


def test_environnement(render: bool, n_steps: int = 300):
    print("\n" + "═" * 55)
    print("  🧪  TEST — DroneAgricoleEnv")
    print("═" * 55)

    env = DroneAgricoleEnv(render_mode="human" if render else None)
    print("  ▶  Reset …")
    obs, info = env.reset()
    print(f"  ✅  Reset OK — obs shape: {obs.shape}")
    print(f"      Position initiale : x={obs[0]:.2f}  y={obs[1]:.2f}  z={obs[2]:.2f}")

    print(f"\n  ▶  {n_steps} pas avec poussée forte (action[0]=0.6, pas de rotation)")
    print("      → Le drone doit monter tout droit\n")

    altitudes = []
    for step in range(n_steps):
        # Poussée à 80% max, zéro rotation — teste juste si le drone peut décoller
        action = np.array([0.6, 0.0, 0.0, 0.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        altitudes.append(float(obs[2]))

        if render:
            time.sleep(1.0 / 240.0)

        if terminated or truncated:
            print(f"  ⚠️  Épisode terminé au step {step} "
                  f"— crash={info.get('crash')}")
            break

        if step % 60 == 0:
            print(f"  step {step:4d}  |  z={obs[2]:6.2f} m  |  "
                  f"reward={reward:+.3f}  |  "
                  f"roll={obs[3]:.2f} rad  pitch={obs[4]:.2f} rad")

    env.close()

    # ── Diagnostic ───────────────────────────────────────────────────────────
    z_max   = max(altitudes) if altitudes else 0.
    z_final = altitudes[-1]  if altitudes else 0.

    print("\n" + "─" * 55)
    print(f"  Altitude max atteinte : {z_max:.2f} m")
    print(f"  Altitude finale       : {z_final:.2f} m")

    if z_max < 0.5:
        print("\n  ❌  PROBLÈME : le drone n'a PAS décollé.")
        print("      Vérifier FORCE_MAX vs masse URDF.")
        print("      Vérifier que WORLD_FRAME est utilisé dans applyExternalForce.")
    elif z_max < 3.0:
        print("\n  ⚠️  Le drone décolle légèrement mais reste bas.")
        print("      Augmente FORCE_MAX dans drone_env.py ou vérifie la masse URDF.")
    else:
        print("\n  ✅  Le drone décolle correctement !")
        print("      Tu peux lancer l'entraînement avec :")
        print("      python train_drl.py --steps 500000")

    print("═" * 55 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui",   action="store_true", help="Fenêtre PyBullet")
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()
    test_environnement(render=args.gui, n_steps=args.steps)
