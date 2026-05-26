# Agente Connect-4 — Híbrido ADP + MCTS

**Fundamentos de Inteligencia Artificial · Universidad de La Sabana · 2026.1**

---

## ¿En qué consiste el agente?

Mi agente combina dos técnicas de IA que vimos en clase: **Adaptive Dynamic Programming (ADP)** y **Monte Carlo Tree Search (MCTS)**, integradas bajo la arquitectura de **Iteración de Política General (GPI)**. La idea principal es que el agente no solo piensa en el turno actual (MCTS), sino que también *aprende y recuerda* de partidas anteriores (ADP), y usa ese conocimiento para tomar mejores decisiones con el tiempo.

Lo que lo diferencia de los otros agentes del grupo es esta combinación: en vez de usar MCTS puro o solo heurísticas, el conocimiento acumulado del ADP le sirve al MCTS como punto de partida, haciendo que las simulaciones sean más eficientes conforme el agente gana experiencia.

---

## Arquitectura

```
┌─────────────────────────────────────────────┐
│         GPI — Iteración de Política         │
│                                             │
│   ┌─────────────┐   prior Q(s,a)            │
│   │     ADP     │ ──────────────────►       │
│   │  Q[s][a]    │                    │      │
│   │  N[s][a]    │◄─── actualización  │      │
│   └─────────────┘     por turno      ▼      │
│                              ┌─────────────┐│
│                              │    MCTS     ││
│                              │    UCB1     ││
│                              │  + rollout  ││
│                              └─────────────┘│
└─────────────────────────────────────────────┘
```

### Componente ADP
Mantiene una tabla `Q[s][a]` que se actualiza con media incremental después de cada movimiento. Funciona como la "memoria" del agente — aprende qué acciones fueron buenas o malas en cada estado.

### Componente MCTS
En cada turno lanza simulaciones dentro del tiempo límite usando **UCB1** para balancear exploración y explotación. Los rollouts usan una heurística simple: primero intenta ganar, luego bloquea al rival, y si no hay ninguna de las dos, juega hacia el centro.

### Integración GPI
Cuando MCTS expande un nodo nuevo, inicializa su valor con `Q(s,a) × 0.3` si el ADP ya tiene experiencia en ese estado. Esto dirige la búsqueda hacia acciones históricamente buenas desde el primer rollout.

---

## Archivos

```
📁 TuNombre/
├── policy.py       # Código del agente
├── entrega.ipynb   # Análisis completo con experimentos y gráficas
└── README.md       # Este archivo
```

---

## Cómo usar el agente

```python
from policy import HybridADPMCTS

agente = HybridADPMCTS(time_limit=1.0)  # 1 segundo por turno
agente.mount()

# En cada turno pasarle el tablero como np.ndarray (6x7)
accion = agente.act(board)
```

### Parámetros configurables

| Parámetro | Default | Descripción |
|---|---|---|
| `time_limit` | `1.0` | Segundos disponibles por turno para las simulaciones MCTS |
| `UCB_C` | `1.41` | Constante de exploración UCB1 (≈ √2) |
| `ADP_WEIGHT` | `0.3` | Peso del prior ADP al inicializar nodos MCTS |

---

## Resultados

- Gana al jugador aleatorio en **+95%** de los casos jugando como rojo y como amarillo.
- Nunca pierde contra el jugador aleatorio.
- Ver análisis completo y gráficas en `entrega.ipynb`.

---

## Dependencias

Solo librerías estándar de Python:
```
numpy
math
time
collections
```
