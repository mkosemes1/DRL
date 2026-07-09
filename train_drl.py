"""
=============================================================================
  TRAIN_DRL.PY  —  Entraînement PPO pour Drone Agricole Hexacoptère
  DIT Dakar — Projet DRL Drone Agricole
=============================================================================
Utilisation :
  python train_drl.py                    # Entraînement complet
  python train_drl.py --resume           # Reprendre depuis checkpoint
  python train_drl.py --eval             # Évaluation GUI (après entraînement)
  python train_drl.py --steps 500000     # Nombre de pas custom
"""

import os
import sys
import argparse
import json
import time
import numpy as np
from datetime import datetime
from pathlib import Path

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import (
        CheckpointCallback, EvalCallback, CallbackList, BaseCallback
    )
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
except ImportError:
    print("❌  stable-baselines3 non installé.")
    print("    pip install stable-baselines3[extra]")
    sys.exit(1)

try:
    import torch
except ImportError:
    print("❌  PyTorch non installé.")
    sys.exit(1)

from drone_env import DroneAgricoleEnv, MAX_STEPS, GAMMA

# ─────────────────────────── Chemins ──────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
LOG_DIR    = BASE_DIR / "logs"
CKPT_DIR   = BASE_DIR / "checkpoints"
MODEL_DIR  = BASE_DIR / "models"
STATS_FILE = BASE_DIR / "training_stats.json"

for d in [LOG_DIR, CKPT_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────── Hyperparamètres PPO ──────────────────────────────
# Repris des conventions "classiques" de Schulman et al. (2017) pour le
# contrôle continu (mêmes ordres de grandeur que les benchmarks MuJoCo cités
# dans Hands-On Modern RL, chap. 7 "PPO" et §12.2.9 — clip_range=0.2,
# gae_lambda=0.95, réseaux [256,256] tanh, Adam lr=3e-4).
#
# Correctif important : l'ancien code utilisait
#   net_arch=[dict(pi=[256,256], vf=[256,256])]   (ancienne API SB3 < 1.8)
# Stable-Baselines3 >= 2.0 attend un dict nu, pas une liste contenant un dict :
#   net_arch=dict(pi=[256,256], vf=[256,256])
# L'ancienne forme est encore tolérée par une couche de compatibilité (avec un
# UserWarning de dépréciation) mais n'est plus documentée ni garantie dans les
# versions futures de SB3 — corrigé ici pour suivre l'API actuelle sans avertissement.
PPO_HPARAMS = {
    "policy":        "MlpPolicy",
    "policy_kwargs": {"net_arch": dict(pi=[256, 256], vf=[256, 256]),
                      "activation_fn": torch.nn.Tanh},
    "n_steps":       2048,
    "batch_size":    256,
    "n_epochs":      10,
    "gamma":         GAMMA,      # importé de drone_env.py : DOIT être identique au γ
                                 # utilisé dans le shaping PBRS (théorème de Ng et al.)
    "gae_lambda":    0.95,      # GAE (Schulman et al., 2016) : compromis biais/variance
    "learning_rate": 3e-4,
    "clip_range":    0.2,       # clipping PPO standard (Schulman et al., 2017)
    "ent_coef":      0.01,
    "vf_coef":       0.5,
    "max_grad_norm": 0.5,
    "target_kl":     0.03,      # garde-fou classique : stoppe une epoch si la
                                 # politique dérive trop (KL > seuil), évite les
                                 # mises à jour destructrices même sous le clipping
    "verbose":       1,
    "tensorboard_log": str(LOG_DIR),
}


# ─────────────────────────── Callback : stats JSON pour dashboard ──────────────
class StatsDashboardCallback(BaseCallback):
    def __init__(self, save_path, verbose=0):
        super().__init__(verbose)
        self.save_path   = save_path
        self._ep_rewards = []
        self._ep_lengths = []
        self._rangees    = []
        self._crashes    = []
        self._sante_fin  = []   # santé globale du champ à la fin de chaque épisode
        self._last_write = time.time()
        self._last_info  = {}

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._ep_rewards.append(info["episode"]["r"])
                self._ep_lengths.append(info["episode"]["l"])
                self._sante_fin.append(info.get("sante_globale", 0.0))
            if "rangees_finies" in info:
                self._rangees.append(info["rangees_finies"])
                self._crashes.append(int(info.get("crash", False)))
                self._last_info = {k: v for k, v in info.items()
                                   if isinstance(v, (int, float, bool))}
        if time.time() - self._last_write > 5.0:
            self._flush()
        return True

    def _flush(self):
        self._last_write = time.time()
        stats = {
            "timestamp":       datetime.now().isoformat(),
            "total_timesteps": int(self.num_timesteps),
            "n_episodes":      len(self._ep_rewards),
            "mean_reward":     float(np.mean(self._ep_rewards[-100:])) if self._ep_rewards else 0.,
            "std_reward":      float(np.std(self._ep_rewards[-100:]))  if self._ep_rewards else 0.,
            "max_reward":      float(np.max(self._ep_rewards))          if self._ep_rewards else 0.,
            "mean_ep_length":  float(np.mean(self._ep_lengths[-100:])) if self._ep_lengths else 0.,
            "mean_rangees":    float(np.mean(self._rangees[-100:]))    if self._rangees    else 0.,
            "crash_rate":      float(np.mean(self._crashes[-100:]))    if self._crashes    else 0.,
            "mean_sante_finale": float(np.mean(self._sante_fin[-100:])) if self._sante_fin  else 0.,
            "rewards_history": [round(r, 2) for r in self._ep_rewards[-200:]],
            "rangees_history": [int(r)       for r in self._rangees[-200:]],
            "sante_history":   [round(s, 4)  for s in self._sante_fin[-200:]],
            "last_info":       self._last_info,
        }
        try:
            with open(self.save_path, "w") as f:
                json.dump(stats, f, indent=2)
        except Exception as e:
            if self.verbose:
                print(f"[StatsCB] Erreur : {e}")

    def _on_training_end(self):
        self._flush()


# ─────────────────────────── Callback : barre console ─────────────────────────
class ConsoleCallback(BaseCallback):
    def __init__(self, total_steps, verbose=1):
        super().__init__(verbose)
        self.total_steps = total_steps
        self._rewards    = []
        self._last_print = 0

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._rewards.append(info["episode"]["r"])
        if self.num_timesteps - self._last_print >= 10_000:
            self._last_print = self.num_timesteps
            pct = 100 * self.num_timesteps / self.total_steps
            rew = np.mean(self._rewards[-50:]) if self._rewards else 0.
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"\r  [{bar}] {pct:5.1f}%  |  "
                  f"Steps: {self.num_timesteps:>8,}  |  "
                  f"Récompense moy: {rew:+8.2f}", end="", flush=True)
        return True


# ─────────────────────────── Entraînement ─────────────────────────────────────
def make_env(render_mode=None, urdf_path=None):
    def _init():
        env = DroneAgricoleEnv(render_mode=render_mode, urdf_path=urdf_path)
        return Monitor(env)
    return _init


def entrainer(total_steps=1_000_000, n_envs=1, resume=False,
              urdf_path=None, render_mode=None):
    print("\n" + "═" * 60)
    print("  🚁  DRL Drone Agricole — Entraînement PPO")
    print("═" * 60)
    print(f"  Timesteps cibles : {total_steps:,}")
    print(f"  Environnements   : {n_envs}")
    print(f"  Device           : {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("═" * 60 + "\n")

    if n_envs == 1:
        env = make_vec_env(make_env(render_mode=render_mode, urdf_path=urdf_path), n_envs=1)
    else:
        env = SubprocVecEnv([make_env(urdf_path=urdf_path) for _ in range(n_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    eval_env = make_vec_env(make_env(urdf_path=urdf_path), n_envs=1)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                            training=False, clip_obs=10.)

    ckpt_path = CKPT_DIR / "drone_ppo_last.zip"
    vec_stats  = CKPT_DIR / "vec_normalize.pkl"

    if resume and ckpt_path.exists():
        print(f"  ✅  Reprise depuis : {ckpt_path}")
        model = PPO.load(str(ckpt_path), env=env)
        if vec_stats.exists():
            env = VecNormalize.load(str(vec_stats), env.venv)
    else:
        print("  🆕  Nouveau modèle PPO")
        model = PPO(env=env, **PPO_HPARAMS)

    callbacks = CallbackList([
        CheckpointCallback(save_freq=50_000, save_path=str(CKPT_DIR),
                           name_prefix="drone_ppo", save_vecnormalize=True),
        EvalCallback(eval_env, best_model_save_path=str(MODEL_DIR),
                     log_path=str(LOG_DIR / "eval"), eval_freq=25_000,
                     n_eval_episodes=3, deterministic=True, verbose=0),
        StatsDashboardCallback(save_path=STATS_FILE),
        ConsoleCallback(total_steps=total_steps),
    ])

    print("  ▶  Démarrage …\n")
    t0 = time.time()
    try:
        model.learn(total_timesteps=total_steps, callback=callbacks,
                    reset_num_timesteps=not resume,
                    tb_log_name=f"PPO_{datetime.now().strftime('%Y%m%d_%H%M')}")
    except KeyboardInterrupt:
        print("\n\n  ⏸  Interrompu.")

    final = MODEL_DIR / "drone_ppo_final"
    model.save(str(final))
    env.save(str(CKPT_DIR / "vec_normalize.pkl"))
    print(f"\n\n  ✅  Modèle sauvegardé : {final}.zip")
    print(f"  ⏱  Durée : {(time.time()-t0)/60:.1f} min\n")
    return model, env


# ─────────────────────────── Évaluation ───────────────────────────────────────
def evaluer(model_path=None, n_episodes=3, urdf_path=None):
    """
    Évaluation avec GUI PyBullet.
    CORRIGÉ : DummyVecEnv obligatoire + VecNormalize rechargé + device=cpu
    """
    mp = model_path or str(MODEL_DIR / "best_model.zip")
    if not os.path.exists(mp):
        if os.path.exists(mp + ".zip"):
            mp = mp + ".zip"
        else:
            print(f"❌  Modèle introuvable : {mp}")
            sys.exit(1)

    vec_stats = CKPT_DIR / "vec_normalize.pkl"
    print(f"\n  🔬  Évaluation : {mp}")

    raw_env = DroneAgricoleEnv(render_mode="human", urdf_path=urdf_path)
    vec_env = DummyVecEnv([lambda: Monitor(raw_env)])

    if vec_stats.exists():
        vec_env = VecNormalize.load(str(vec_stats), vec_env)
        vec_env.training    = False
        vec_env.norm_reward = False
        print("  ✅  Normalisation rechargée.")
    else:
        print("  ⚠️  VecNormalize absent — résultats potentiellement dégradés.")

    # device="cpu" : évite le bug driver CUDA trop vieux
    model = PPO.load(mp, env=vec_env, device="cpu")
    print("  ✅  Modèle chargé sur CPU\n")

    for ep in range(n_episodes):
        obs       = vec_env.reset()
        done      = False
        total     = 0.
        step      = 0
        last_info = {}

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = vec_env.step(action)
            total    += float(reward[0])
            step     += 1
            done      = bool(dones[0])
            last_info = infos[0] if infos else {}
            time.sleep(1. / 60.)

        print(f"  Épisode {ep+1}/{n_episodes}  |  "
              f"Récompense: {total:+.2f}  |  "
              f"Rangées: {last_info.get('rangees_finies','?')}/4  |  "
              f"Crash: {last_info.get('crash','?')}  |  "
              f"Masse: {last_info.get('masse','?')} kg  |  "
              f"Steps: {step}")

    vec_env.close()
    print("\n  ✅  Évaluation terminée.")


# ─────────────────────────── Point d'entrée ───────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement DRL — Drone Agricole")
    parser.add_argument("--steps",      type=int, default=1_000_000)
    parser.add_argument("--n-envs",     type=int, default=1)
    parser.add_argument("--resume",     action="store_true")
    parser.add_argument("--eval",       action="store_true")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--urdf-path",  type=str, default="models/agri_hexacopter.urdf")
    parser.add_argument("--env-render", type=str, default=None, choices=["human"])
    parser.add_argument("--n-eval-eps", type=int, default=3)
    args = parser.parse_args()

    if args.eval:
        evaluer(model_path=args.model_path, n_episodes=args.n_eval_eps,
                urdf_path=args.urdf_path)
    else:
        entrainer(total_steps=args.steps, n_envs=args.n_envs, resume=args.resume,
                  urdf_path=args.urdf_path, render_mode=args.env_render)
