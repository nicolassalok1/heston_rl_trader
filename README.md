# Heston-RL-Trader

Framework complet pour construire un agent de trading en Reinforcement Learning exploitant :

- Un inverseur Heston (CNN + reconstruction) pour extraire des régimes volatiles,
- Un module Shitcoin basé sur des pseudo-surfaces (moments glissants),
- Un module BTC basé sur une surface IV complète,
- Une architecture FeatureEngine → StateBuilder → RL Agent,
- Un environnement Gym de trading,
- Un PPO agent optimisé pour signaux régimes-volatilité.

Ce projet donne une architecture complète prête à raccorder à vos données réelles
(perps, futures, orderbook, options Deribit, sentiment Telegram/Twitter).




Quand tu passeras aux vraies surfaces IV (Deribit) et aux vraies séries spot (Binance) :

tu te sers du même modèle (pré-entraîné synthétiquement),

tu lui fais un fine-tuning sur tes vraies surfaces,

ton dataset réel doit juste :

ré-échantillonner sur une grille 
(
𝑘
,
𝑇
)
(k,T) cohérente,

forward-fill / drop les points manquants,

aligner les timestamps (timezone → UTC, puis offset Europe/Paris si tu veux visualiser).