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
