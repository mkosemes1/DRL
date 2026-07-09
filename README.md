# 🚁 DRL — Drone Agricole Autonome (Hexacoptère)

Projet de fin d'année **L2 Big Data **.

Un hexacoptère agricole apprend, par **apprentissage par renforcement profond (PPO)**,
à survoler un champ de 4 rangées de cultures, à s'aligner et voler à l'altitude de
travail, à pulvériser sa charge utile au bon endroit, puis à terminer chaque rangée
jusqu'à la fin de la mission — sans intervention humaine, uniquement à partir d'un
signal de récompense.

Le simulateur physique est [PyBullet](https://pybullet.org/), l'algorithme d'apprentissage
est **PPO (Proximal Policy Optimization)** via [Stable-Baselines3](https://stable-baselines3.readthedocs.io/).

---

## Sommaire

1. [Ce qui a été refait, et pourquoi](#1-ce-qui-a-été-refait-et-pourquoi)
2. [Architecture du projet](#2-architecture-du-projet)
3. [L'environnement `DroneAgricoleEnv`](#3-lenvironnement-droneagricoleenv)
4. [Conception de la récompense (le cœur de la refonte)](#4-conception-de-la-récompense-le-cœur-de-la-refonte)
   - [4.6 Santé des plantes et pulvérisation ciblée](#46-santé-des-plantes-et-pulvérisation-ciblée-v5)
5. [L'algorithme PPO utilisé](#5-lalgorithme-ppo-utilisé)
6. [Installation](#6-installation)
7. [Utilisation](#7-utilisation)
8. [Dashboard de supervision](#8-dashboard-de-supervision)
9. [Limites connues et pistes d'amélioration](#9-limites-connues-et-pistes-damélioration)
10. [Références](#10-références)

---

## 1. Ce qui a été refait, et pourquoi

La demande initiale de ce projet était de **garantir la cohérence et la légitimité
de la récompense**, en suivant les méthodes classiques de PPO. Un audit du code
existant (`drone_env.py` v3) a révélé un problème classique et bien documenté en RL,
que ce README explique en détail (§4) : **la fonction de récompense additionnait,
à chaque pas de temps, plusieurs bonus denses inconditionnels** (survie, altitude,
alignement, avancement) dont la somme sur un épisode dépassait largement les
récompenses "de mission" (finir une rangée, finir la mission). Concrètement, un
agent pouvait accumuler plus de récompense en **planant sagement dans le champ**
qu'en **finissant réellement la mission** — un cas d'école de *reward hacking* /
désalignement de la récompense.

**Ce qui a été changé :**

| Fichier | Changement |
|---|---|
| `drone_env.py` | Fonction de récompense entièrement reconstruite selon le *reward shaping potential-based* (PBRS, Ng, Harada & Russell 1999) : un seul terme de shaping dense, mathématiquement borné (au sens actualisé), + des récompenses éparses (sparse) pour les vrais objectifs de la tâche. Suppression du bonus de survie inconditionnel. `γ` défini une seule fois ici (`GAMMA=0.9995`) et importé par `train_drl.py`, pour respecter l'exigence du théorème PBRS d'un `γ` identique entre l'environnement et l'agent. **v5** : la pulvérisation récompense désormais la santé *réellement restaurée* par plante (simulation de vigueur type NDVI, 120 plantes), pas juste la masse dépensée — voir §4.6. |
| `train_drl.py` | Correction de `net_arch` (ancienne API SB3 dépréciée) ; ajout de `target_kl` (garde-fou standard de PPO) ; `gamma` importé de `drone_env.py` au lieu d'être dupliqué en dur, et relevé de 0.995 à 0.9995 (voir §4.5 — l'ancienne valeur rendait le retour actualisé d'une mission réussie *inférieur* à celui d'un drone immobile). |
| `requirements.txt` | Ajout de `stable-baselines3`, `pandas`, `plotly`, `streamlit` — importés dans le projet mais absents du fichier, ce qui aurait empêché `train_drl.py` et `supervision.py` de s'exécuter. Trois versions ré-épinglées pour lever des conflits de résolution avec `streamlit` (`protobuf` 7.34.1→5.29.6, `packaging` 26.2→24.2, `rich` 15.0.0→13.9.4) — installation vérifiée avec `pip install --dry-run`, aucun conflit restant. |

Le reste de l'architecture (physique du drone, détection de crash, génération du
champ 3D, dashboard Streamlit) était déjà solide et a été conservé tel quel.

---

## 2. Architecture du projet

```
DRL/
├── drone_env.py              # Environnement Gymnasium (physique + récompense) — cœur du RL
├── train_drl.py               # Entraînement / évaluation PPO (Stable-Baselines3)
├── test_env.py                # Test rapide de la physique (décollage) avant entraînement
├── main.py                    # Démo brute PyBullet (chargement URDF, sans RL)
├── environnement_agricole.py  # Ancien script de démonstration du champ (non-RL, legacy)
├── pilotage.py                # Ancien script de pilotage scripté avec logs CSV (non-RL, legacy)
├── supervision.py             # Dashboard Streamlit — suit l'entraînement en temps réel
├── models/
│   └── agri_hexacopter.urdf   # Modèle physique du drone (corps, bras, rotors)
├── data/                      # Logs et rapports de simulation
├── checkpoints/                # Créé automatiquement pendant l'entraînement
├── logs/                       # Logs TensorBoard + évaluation
└── requirements.txt
```

`main.py`, `environnement_agricole.py` et `pilotage.py` sont des scripts antérieurs
au passage au RL (pilotage scripté ou visualisation brute PyBullet). Ils ne sont
pas utilisés par l'entraînement PPO mais sont conservés pour la démonstration de
la physique et l'historique du projet.

---

## 3. L'environnement `DroneAgricoleEnv`

Environnement `gymnasium.Env` personnalisé, simulé avec PyBullet.

**Tâche :** parcourir 4 rangées de culture (`Y = -4.5, -1.5, 1.5, 4.5`, 30 plantes
chacune), en alternant le sens de déplacement (aller X: -20→+20, retour X:
+20→-20), en maintenant l'altitude de travail (15 m) et l'alignement latéral,
tout en pulvérisant réellement les plantes stressées rencontrées (la masse du
drone diminue au fur et à mesure, jusqu'à une charge utile minimale — voir §4.6
pour le mécanisme de santé des plantes qui détermine si la pulvérisation sert
réellement à quelque chose).

**Espace d'observation** (21 valeurs continues) : position (x, y, z), orientation
(roll, pitch, yaw), vitesse linéaire et angulaire, masse courante, index de la
rangée active, écarts à la cible (x, y, z), état de pulvérisation, **+ 3 signaux
de santé du champ** (santé locale sous le rayon de pulvérisation, santé moyenne
de la rangée cible, santé moyenne globale — voir §4.6).

**Espace d'action** (4 valeurs continues dans `[-1, 1]`) : poussée verticale
totale, couple de roulis, couple de tangage, couple de lacet — un contrôle bas
niveau typique d'un multirotor.

**Fin d'épisode :**
- `terminated` : crash (altitude trop basse ou inclinaison excessive) **ou**
  mission terminée (4 rangées complétées) ;
- `truncated` : `MAX_STEPS = 4000` atteints, ou sortie de la zone de vol autorisée.

La physique (masses, amortissements, force max, couple max, préchauffage au
reset) date de la v3 et a été conservée : elle avait déjà été corrigée pour
stabiliser le décollage et éviter les faux crashes en tout début d'épisode.

---

## 4. Conception de la récompense (le cœur de la refonte)

### 4.1 Le problème de la v3

La v3 empilait, à chaque pas de temps, plusieurs bonus **inconditionnels** :

```
+0.05  si en l'air
+0.10  si altitude correcte
+0.15  si aligné sur la rangée
+0.20 × dx  si progression positive
+0.15  si en train de pulvériser (×3 = 0.15 en réalité, avec le coefficient)
```

Sur un épisode de 4000 pas, un agent qui reste simplement en vol stable, bien
aligné, à la bonne altitude, **sans jamais finir une rangée**, peut accumuler
largement plus de 500 à 1000 de récompense — alors que finir une rangée ne
rapporte que **10**, et finir toute la mission seulement **50**. La politique
optimale au sens de cette fonction de récompense n'est donc **pas** "livrer la
mission", mais "planer sagement le plus longtemps possible".

C'est exactement le piège décrit dans le manuel utilisé comme référence pour ce
projet (*Hands-On Modern RL*, §3.9, "Reward Function Design") : un bonus par pas
de temps sans contrepartie liée à l'achèvement de la tâche pousse l'agent à
boucler indéfiniment près de l'objectif plutôt qu'à le terminer, dès lors qu'il
n'y a pas de mécanisme qui rend cette boucle moins rentable que la réussite.

### 4.2 La solution : reward shaping potential-based (PBRS)

Ng, Harada et Russell (ICML 1999) ont démontré que si l'on ajoute à la récompense
d'origine un terme de la forme

```
F(s, a, s') = γ·Φ(s') − Φ(s)
```

où `Φ` est une **fonction de potentiel qui ne dépend que de l'état** (jamais de
l'action) et où **γ est le même facteur d'actualisation que celui utilisé par
l'agent**, alors **la politique optimale de la tâche d'origine ne change pas**.
Plus précisément, le résultat exact est :

```
V'^π(s) = V^π(s) − Φ(s)      pour toute politique π et tout état s
```

c'est-à-dire que la valeur de chaque politique est décalée d'une **constante qui
ne dépend que de l'état de départ**, jamais de la politique elle-même — donc le
classement des politiques (et en particulier la politique optimale) est
strictement inchangé.

> **Écart corrigé pendant la revue.** Une première version de ce projet
> utilisait `F = Φ(s') − Φ(s)`, **sans le facteur γ**, par analogie avec
> l'exemple BipedalWalker cité dans le manuel de référence (qui fait la même
> simplification et se qualifie lui-même de "PBRS-*like*", pas de PBRS strict).
> Ce n'est pas l'erreur la plus grave qui soit — pour un `γ` proche de 1 l'écart
> est faible — mais ce n'est pas non plus le théorème exact : le théorème exige
> précisément le même `γ` que celui de l'algorithme. La version actuelle du
> code utilise la formule stricte, avec une **constante `GAMMA` unique définie
> dans `drone_env.py`** et importée telle quelle par `train_drl.py` pour
> configurer PPO — ainsi l'environnement et l'algorithme ne peuvent pas
> diverger silencieusement sur cette valeur, ce qui invaliderait la garantie.

**Propriété clé — le télescopage, dans le domaine actualisé.** La somme
*actualisée* des termes de shaping sur un épisode se réduit à :

```
Σₜ γᵗ · (γΦ(sₜ₊₁) − Φ(sₜ)) = γᵀΦ(s_T) − Φ(s₀)
```

C'est cette version *actualisée* du télescopage qui est garantie par le
théorème — pas la somme brute (non actualisée) des récompenses. Concrètement,
cela veut dire que le shaping ne peut jamais, à lui seul, rendre une politique
sous-optimale « artificiellement » meilleure qu'une politique optimale : tout
gain apparent du shaping est un décalage identique pour toutes les politiques
partant du même état.

### 4.3 La fonction de potentiel implémentée

```python
def _potentiel(self, x, y, z, target_x, target_y):
    d_x = abs(target_x - x) / (X_FIN - X_DEBUT)   # avancement le long de la rangée, dans [0,1]
    d_y = abs(y - target_y) / 10.0                # alignement latéral
    d_z = abs(z - ALTITUDE_TRAVAIL) / ALTITUDE_TRAVAIL  # écart à l'altitude de travail
    return -(d_x + 0.6 * d_y + 0.6 * d_z)
```

`Φ(s)` vaut 0 lorsque le drone est exactement à la position cible (fin de
rangée, bon alignement, bonne altitude), et devient de plus en plus négatif à
mesure qu'il s'en éloigne. Le shaping appliqué à chaque pas est
`K_SHAPING × (GAMMA × Φ(s') − Φ(s))`, avec `K_SHAPING = 5.0` et `GAMMA = 0.9995`
(voir §4.5 pour la justification de cette valeur précise de `GAMMA`).

> **Limite assumée** : la cible (`target_x`, `target_y`) change au pas exact où
> une rangée se termine (rangée suivante, sens inversé). Cela crée un saut
> ponctuel de `Φ`, ce qui n'est pas un PBRS "pur" à cet instant précis. C'est un
> compromis documenté : la rangée terminée est de toute façon déjà récompensée
> explicitement par `R_FIN_RANGEE`, donc ce saut ne crée pas d'incitation
> perverse — il ne fait que réinitialiser le repère de progression pour la
> rangée suivante (une forme de curriculum de sous-objectifs, également décrite
> dans le manuel, §7.7.6).

### 4.4 Le budget de récompense complet

| Terme | Type | Valeur | Rôle |
|---|---|---|---|
| `K_SHAPING × (γΦ(s')−Φ(s))` | dense, borné (PBRS strict) | gain 5.0 | signal d'apprentissage continu vers la cible |
| `R_FIN_RANGEE` | sparse | +20 / rangée | objectif intermédiaire réel |
| `R_MISSION_FINIE` | sparse | +100 | objectif final réel |
| `R_SANTE_PAR_UNITE × Δsanté` | sparse, borné physiquement | +1.0 / unité de santé restaurée | pulvérisation ciblée — voir §4.6 |
| `P_CRASH` | sparse, terminal | −100 | échec — même ordre de grandeur que la mission réussie |
| `P_DIVERGENCE` | sparse, terminal | −20 | sortie de la zone de vol |
| `P_INCLINAISON` | dense, faible poids | −0.05 × inclinaison | stabilité (posture) |
| `P_CONTROLE` | dense, faible poids | −0.001 × ‖action‖² | coût de contrôle (style HalfCheetah), évite les commandes brusques |
| `P_TEMPS` | dense, faible poids | −0.002 / pas | encourage l'efficacité, à récompense de tâche égale |

Cette décomposition suit directement les recommandations du manuel de référence
sur la composition de récompenses multi-termes (§3.9.4) : séparer un terme dense
de shaping (justifié théoriquement, PBRS) des récompenses éparses qui définissent
la tâche, et garder les coefficients de coût secondaires (contrôle, temps,
inclinaison) **petits par rapport** aux récompenses de tâche.

### 4.5 Pourquoi `γ = 0.9995` et pas `0.99` ou `0.995`

Le choix de `γ` n'est pas qu'un détail de PPO — c'est lui qui détermine
**l'horizon effectif** sur lequel l'agent valorise le futur, approximativement
`1 / (1 − γ)` pas de temps. Avec `MAX_STEPS = 4000` et des récompenses de tâche
concentrées en fin d'épisode (fin de rangée vers 750-1000 pas, mission complète
vers 3000-4000 pas), un `γ` insuffisant **efface presque totalement** ces
récompenses du retour actualisé calculé depuis le début de l'épisode — même si
la garantie d'invariance du PBRS reste vraie *algébriquement*, elle devient
numériquement inutile si l'horizon de `γ` est plus court que l'horizon réel de
la tâche.

Ceci a été vérifié numériquement (script de simulation, sans PyBullet, retour
actualisé `Σ γᵗ·rₜ` sur des scénarios idéalisés) :

| `γ` | Horizon effectif `1/(1−γ)` | Retour actualisé — mission réussie | Retour actualisé — immobile 4000 pas | Retour actualisé — crash |
|---|---|---|---|---|
| 0.99 | 100 pas | **+4.8** | +7.4 | −55.4 |
| 0.995 | 200 pas | **+5.1** | +7.2 | −74.5 |
| **0.9995 (retenu)** | **2000 pas** | **+47 à +67** | **+3.1** | **−97.2** |

Avec `γ = 0.99` **ou** `γ = 0.995`, le retour actualisé d'une mission réussie
devient **inférieur** à celui d'un drone qui ne bouge pas — la hiérarchie que
toute la conception de la récompense (§4.1–4.4) cherche à garantir se serait
donc effondrée dans les faits, malgré une formule PBRS mathématiquement propre.
`γ = 0.9995` restaure une marge saine et robuste, y compris pour une mission
plus lente qui utiliserait une grande partie des 4000 pas disponibles. C'est un
exemple concret de la mise en garde classique de PPO : `γ` doit être choisi en
fonction de **l'horizon réel de la tâche**, pas fixé par habitude à 0.99.

`gae_lambda = 0.95` reste inchangé : ce paramètre contrôle le compromis
biais/variance de l'estimateur d'avantage (GAE, Schulman et al. 2016) et sa
valeur usuelle reste adaptée indépendamment du choix de `γ`.

### 4.6 Santé des plantes et pulvérisation ciblée (v5)

Jusqu'à la v4, la récompense de pulvérisation ne dépendait que de la **masse
dépensée** : le drone à la bonne altitude et bien aligné sur une rangée
"pulvérisait" et était récompensé, sans que l'environnement vérifie s'il y
avait réellement une plante sous lui, ni si elle en avait besoin. Un agent
aurait donc pu apprendre à survoler n'importe quel point de la rangée pour
être payé — la récompense mesurait un *geste* ("suis-je en train de pulvériser
au sens géométrique ?"), pas un *résultat* ("ai-je réellement soigné quelque
chose ?"). La v5 corrige ce défaut de légitimité.

**Simulation de santé.** Chaque plante (30 par rangée × 4 rangées = 120 au
total, aux mêmes positions que les plantes affichées à l'écran) possède un
état de santé continu `santé ∈ [0, 1]` — un analogue simplifié d'un indice de
vigueur type NDVI. Au reset, chaque plante démarre à une valeur aléatoire
indépendante entre `SANTE_INIT_MIN=0.25` et `SANTE_INIT_MAX=0.65` : le champ
est toujours partiellement stressé au début de la mission, et jamais
totalement sain — il y a donc toujours un vrai travail à faire, différent à
chaque épisode (meilleure généralisation que si le déficit était fixe).

**Traitement.** Quand le drone pulvérise (mêmes conditions géométriques
qu'avant : bonne altitude, bon alignement latéral), chaque plante de la
rangée courante à une distance `d` du drone en x reçoit une dose

```
efficacité = max(0, 1 − d / SPRAY_RADIUS_X)        # 1 au centre du cône, 0 au bord
dose       = TAUX_TRAITEMENT × DT × efficacité
santé      = min(1.0, santé + dose)
```

avec `SPRAY_RADIUS_X = 2.5 m` et `TAUX_TRAITEMENT = 2.0`/s. Ces valeurs ont
été calibrées numériquement (simulation d'un passage rectiligne à vitesse de
croisière ~13 m/s, script indépendant de PyBullet) : un unique passage porte
la santé moyenne d'une rangée de ~0,45 à ~0,82 ; un passage deux fois plus
lent la porte à ~0,98. **C'est un vrai compromis vitesse/qualité** — voler
plus vite couvre plus de terrain par unité de temps mais soigne moins bien
chaque plante, exactement comme un vrai drone agricole doit arbitrer entre
débit de chantier et dose déposée.

**Récompense.** Le terme `r_sante` de l'étape 4.4 vaut `R_SANTE_PAR_UNITE ×
Σ Δsanté`, où `Δsanté` est le gain **réel** de chaque plante effectivement
couverte ce pas-ci (0 si elle est déjà à 1.0, ou si aucune plante n'est dans
le rayon). Comme pour l'ancien mécanisme basé sur la masse, ce terme reste
**intrinsèquement borné** : une fois toutes les plantes à `santé=1.0`, plus
aucune récompense n'en provient, quelle que soit la durée restante de
l'épisode — pas de fuite de récompense possible en repassant sur une zone
déjà traitée.

**Perception.** Un signal de récompense que l'agent ne peut pas observer est
inutile à PPO — la politique n'a aucun levier pour l'optimiser. Trois valeurs
sont donc ajoutées à l'observation (§3, `_get_obs`) :
- `sante_locale` — santé moyenne des plantes actuellement dans le rayon de
  pulvérisation (« y a-t-il du travail juste ici ? ») ;
- `sante_rangee` — santé moyenne de la rangée cible courante (progression sur
  la rangée en cours) ;
- `sante_globale` — santé moyenne des 120 plantes (progression de la mission).

**Rendu visuel.** En `render_mode="human"` uniquement (jamais pendant
l'entraînement headless, pour ne pas ralentir la simulation), les feuilles de
chaque plante sont coloriées selon sa santé — jaune-brun (stressée) à vert
(saine) — et mises à jour en temps réel à mesure qu'elles sont traitées. Le
champ démarre donc visiblement inégal (certaines plantes plus jaunes que
d'autres) et verdit progressivement là où le drone est réellement passé.

**Vérification numérique effectuée pendant le développement** (simulation
complète d'une mission, avec le mécanisme de santé exact ci-dessus, retour
actualisé à `γ=0.9995`) :

| Scénario | Retour actualisé | Santé finale du champ |
|---|---|---|
| Mission complète, vitesse normale | **+81.3** | 0.45 → 0.82 |
| Mission complète, vitesse réduite (2×) | **+66.0** | 0.45 → 0.91 |
| Immobile 4000 pas | +3.1 | inchangée |
| Crash immédiat | −97.2 | inchangée |

La hiérarchie `mission > immobile > crash` tient toujours (voir §4.5), et le
mécanisme produit en plus un vrai signal d'arbitrage vitesse/qualité — vérifié
également par un test fonctionnel qui exécute réellement `reset()`/`step()`
(avec un stub remplaçant PyBullet, faute de pouvoir compiler la dépendance
native dans l'environnement de développement utilisé pour cette revue) : un
drone immobile au-dessus de la première rangée traite exactement les 2 plantes
dans son rayon de 2,5 m et aucune autre, avec une récompense qui correspond au
calcul attendu à 4 décimales près.

> **Limite assumée.** Ceci reste une simulation simplifiée de l'état des
> cultures (un scalaire par plante, sans modèle agronomique de maladie, de
> propagation, ni de dépendance à la météo ou au type de culture). Ce n'est
> pas une analyse de santé par vision embarquée (pas de caméra, pas de calcul
> d'indice NDVI à partir d'images) — c'est un état interne de l'environnement,
> comparable à un capteur au sol qui donnerait un score de stress. Un travail
> de vision par ordinateur (segmentation d'image, calcul d'indices spectraux)
> serait un projet à part entière, hors du périmètre de cette correction.

---

## 5. L'algorithme PPO utilisé

**PPO (Proximal Policy Optimization)**, via `stable_baselines3.PPO`, avec les
hyperparamètres suivants (`train_drl.py`, dict `PPO_HPARAMS`) :

| Hyperparamètre | Valeur | Justification |
|---|---|---|
| Réseau (policy / value) | `[256, 256]`, activation `Tanh` | architecture standard pour du contrôle continu (benchmarks MuJoCo) |
| `n_steps` | 2048 | taille de rollout classique pour PPO continu |
| `batch_size` | 256 | 8 mini-batchs par rollout |
| `n_epochs` | 10 | nombre de passes d'optimisation par rollout (valeur du papier original) |
| `gamma` | **0.9995** *(importé de `drone_env.py`, corrigé — voir §4.5)* | horizon effectif ≈2000 pas, cohérent avec `MAX_STEPS=4000` et le PBRS |
| `gae_lambda` | 0.95 | *Generalized Advantage Estimation* (Schulman et al., 2016), compromis biais/variance |
| `clip_range` | 0.2 | clipping standard de l'objectif PPO (Schulman et al., 2017) |
| `clip_range_vf` | non défini (`None`, défaut SB3) | fidèle à l'algorithme original du papier — le clipping de la value function est un ajout d'OpenAI Baselines, pas du cœur de PPO |
| `normalize_advantage` | non défini (`True`, défaut SB3) | normalisation de l'avantage par mini-batch, pratique standard qui stabilise le gradient de politique |
| `ent_coef` | 0.01 | bonus d'entropie, maintient l'exploration |
| `vf_coef` | 0.5 | poids de la perte de la fonction de valeur |
| `max_grad_norm` | 0.5 | *gradient clipping*, stabilité de l'optimisation |
| `target_kl` | 0.03 *(ajouté)* | garde-fou : arrête une epoch si la politique dérive trop, évite les mises à jour destructrices |
| `learning_rate` | 3e-4 | valeur standard Adam pour PPO |

Les deux lignes `clip_range_vf` et `normalize_advantage` ne sont pas explicitement
présentes dans `PPO_HPARAMS` : leur valeur par défaut a été **vérifiée
directement dans le code source** de `stable_baselines3==2.9.0` installé (et
non supposée de mémoire), pour confirmer qu'elle correspond bien aux
conventions classiques avant de les laisser implicites.

**Correction de compatibilité :** l'ancienne configuration passait
`net_arch=[dict(pi=[256,256], vf=[256,256])]` (liste contenant un dict), une
API dépréciée depuis Stable-Baselines3 1.8 qui déclenche un avertissement de
dépréciation et n'est plus documentée. La version actuelle passe directement
`net_arch=dict(pi=[256,256], vf=[256,256])`, conforme à l'API SB3 ≥ 2.0
(vérifié avec `stable-baselines3==2.9.0`, la version épinglée dans
`requirements.txt`).

**Normalisation.** L'entraînement utilise `VecNormalize` (observations et
récompenses), ce qui est la pratique standard pour stabiliser PPO sur des
environnements physiques où les échelles des observations (position en mètres,
angles en radians, vitesses...) diffèrent fortement. À l'évaluation,
`norm_reward=False` et `training=False` évitent de fausser les récompenses
rapportées — c'était déjà correctement géré dans le code existant.

---

## 6. Installation

```bash
cd DRL
python3 -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

> `requirements.txt` épingle des versions exactes (générées sur l'environnement
> de développement d'origine). Si votre machine n'a pas de GPU CUDA compatible,
> PyTorch retombera automatiquement sur CPU — l'entraînement fonctionne, juste
> plus lentement.

**Vérifier que la physique fonctionne avant de lancer un entraînement long :**

```bash
python test_env.py --gui --steps 300
```

Le drone doit décoller et monter droit. Le script affiche un diagnostic clair
si ce n'est pas le cas (force insuffisante, masse mal configurée, etc.).

---

## 7. Utilisation

### Entraîner

```bash
python train_drl.py --steps 1000000
```

- `--steps N` : nombre total de pas d'entraînement (défaut : 1 000 000).
- `--n-envs N` : nombre d'environnements parallèles (défaut : 1 ; `SubprocVecEnv`
  si `N > 1`, utile si vous avez plusieurs cœurs CPU disponibles).
- `--resume` : reprendre depuis le dernier checkpoint (`checkpoints/drone_ppo_last.zip`).
- `--urdf-path` : chemin vers un autre modèle URDF de drone.

Pendant l'entraînement :
- des checkpoints sont sauvegardés tous les 50 000 pas dans `checkpoints/` ;
- le meilleur modèle (au sens de l'évaluation périodique) est sauvegardé dans
  `models/best_model.zip` ;
- des statistiques (récompense moyenne, taux de crash, rangées finies...) sont
  écrites en continu dans `training_stats.json`, consommées par le dashboard
  Streamlit (§8) ;
- les logs TensorBoard sont dans `logs/` (`tensorboard --logdir logs`).

### Évaluer (avec rendu graphique)

```bash
python train_drl.py --eval --n-eval-eps 5
```

Charge `models/best_model.zip` (ou `--model-path` pour un autre fichier), recharge
les statistiques de normalisation (`checkpoints/vec_normalize.pkl`), et affiche
la simulation en temps réel dans une fenêtre PyBullet.

### Test rapide de la physique seule

```bash
python test_env.py            # sans fenêtre
python test_env.py --gui      # avec fenêtre PyBullet
```

---

## 8. Dashboard de supervision

```bash
streamlit run supervision.py
```

Lit `training_stats.json` en continu pendant l'entraînement et affiche :
récompense moyenne / écart-type, taux de crash, nombre moyen de rangées
finies, courbes d'évolution. Utile pour suivre l'entraînement sans dépendre de
TensorBoard.

---

## 9. Limites connues et pistes d'amélioration

- **Discontinuité de `Φ` aux transitions de rangée** (voir §4.3) : acceptable
  ici car compensée par une récompense sparse dédiée, mais une version plus
  rigoureuse pourrait utiliser un potentiel global unique (ex. distance le long
  d'un chemin serpentin prédéfini) pour un PBRS parfaitement continu sur toute
  la mission.
- **`n_envs=1` par défaut** : l'entraînement est plus rapide avec plusieurs
  environnements PyBullet en parallèle (`--n-envs 4` ou plus sur une machine
  multi-cœurs) — PyBullet en mode `DIRECT` (sans rendu) le permet.
- **Pas de curriculum explicite** : le potentiel PBRS densifie déjà
  l'apprentissage, mais un curriculum qui commencerait par une seule rangée
  courte avant d'introduire les 4 rangées complètes pourrait accélérer la
  convergence sur du matériel limité (cf. manuel, §7.7.6, "Curriculum
  Learning").
- **Aucun domain randomization** : la masse, la friction, le vent ne sont pas
  randomisés à l'entraînement, ce qui limiterait la robustesse en cas de
  transfert vers un drone réel (*sim-to-real gap*, manuel §12.2).
- **Santé des plantes simplifiée** (voir §4.6) : un scalaire par plante sans
  vision embarquée ni modèle agronomique — suffisant pour rendre la
  pulvérisation légitime au sein de l'environnement RL, mais pas un substitut
  à une vraie chaîne de perception (caméra + calcul d'indice type NDVI).

---

## 10. Références

- Ng, A. Y., Harada, D., & Russell, S. (1999). *Policy invariance under reward
  transformations: Theory and application to reward shaping.* ICML.
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017).
  *Proximal Policy Optimization Algorithms.* arXiv:1707.06347.
- Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2016).
  *High-Dimensional Continuous Control Using Generalized Advantage Estimation.*
  arXiv:1506.02438.
- WalkingLabs, *Hands-On Modern RL* (manuel fourni pour ce projet) — en
  particulier §3.9 "Reward Function Design", §7 "PPO", et §12.2.9 "Hands-On:
  Training a Simulated Robot to Run with PPO".
- [Stable-Baselines3 — documentation officielle](https://stable-baselines3.readthedocs.io/)
- [PyBullet Quickstart Guide](https://pybullet.org/)
