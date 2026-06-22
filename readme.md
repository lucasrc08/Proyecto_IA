# Agente Connect-4 — MCTS con UCB1

**Fundamentos de Inteligencia Artificial — Universidad de La Sabana 2026.1**

---

## Descripción del agente

El agente implementa **Monte Carlo Tree Search (MCTS)** con política de selección **UCB1**. En cada turno construye un árbol de búsqueda desde el estado actual del tablero y elige la columna cuyo nodo hijo fue más visitado al agotar el presupuesto de tiempo.

Las cuatro fases del algoritmo son:

1. **Selección** — baja el árbol eligiendo siempre el hijo con mayor UCB1: `wins/visits + C·√(ln(N)/n)`, balanceando explotación y exploración.
2. **Expansión** — agrega un nodo hijo no explorado, priorizando columnas del centro (col 3 > col 2/4 > col 1/5 > col 0/6).
3. **Simulación** — juega hasta el final con lógica de prioridad: victoria inmediata → bloqueo al rival → aleatorio ponderado al centro.
4. **Retropropagación** — actualiza `wins` y `visits` en cada nodo ancestro, invirtiendo el resultado al cambiar de perspectiva entre jugadores.

El agente infiere su color contando fichas en el tablero (no requiere que se lo pasen explícitamente).

---

## Archivos del agente

| Archivo | Descripción |
|---------|-------------|
| `policy.py` | Código completo del agente (`MCTSPolicy`) |
| `entrega.ipynb` | Análisis experimental: comparación v1 vs v2, curva de iteraciones, rendimiento por color y propuestas de mejora |
| `readme.md` | Este archivo |

No se requieren archivos adicionales (modelos, pesos, tablas). El agente razona en tiempo real sin fase de entrenamiento offline.

---

## Dependencias

- Python 3.8 o superior
- `numpy`
- `matplotlib` (solo para `entrega.ipynb`)

Todas vienen incluidas en Anaconda. No se necesita instalar nada adicional.

---

## Integración con el torneo

Colocar `policy.py` en la carpeta del grupo dentro del torneo:

```
tournament/
├── connect4/          ← del zip del torneo
├── groups/
│   └── <nombre>/
│       └── policy.py  ← aquí
├── policy.py          ← también en la raíz para pruebas locales
└── main.py
```

El torneo llama automáticamente a `MCTSPolicy.mount(time_limit)` antes de cada partida y a `MCTSPolicy.act(board)` en cada turno. No se requiere ninguna otra configuración.

---

## Parámetros configurables

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `time_limit` | `1.0` s | Segundos disponibles por turno. El agente usa el 80% como margen de seguridad. |
| `iterations` | `None` | Si se especifica, ignora `time_limit` y hace exactamente N simulaciones por turno. |
| `exploration_c` | `√2` | Constante de exploración de UCB1. Valores mayores exploran más; menores explotan más. |

Ejemplo de uso directo:

```python
from policy import MCTSPolicy

agent = MCTSPolicy(time_limit=1.0)
agent.mount()
columna = agent.act(board)   # board: np.ndarray shape (6,7), -1=Rojo, 1=Amarillo, 0=vacío
```

---

## Cómo correr el análisis (`entrega.ipynb`)

1. Descomprimir el zip del torneo.
2. Copiar `policy.py` y `entrega.ipynb` en la raíz de la carpeta `tournament/`.
3. Abrir Jupyter desde esa carpeta (Anaconda Navigator → JupyterLab o `jupyter notebook`).
4. Abrir `entrega.ipynb` y ejecutar **Run All Cells**.

La estructura esperada antes de correr:

```
tournament/
├── connect4/       ← necesario para los imports del notebook
├── policy.py       ← el agente
└── entrega.ipynb   ← el análisis
```

Tiempo estimado de ejecución: **~35 minutos** con la configuración por defecto (20 partidas por condición).

---

## Resultados principales

| Experimento | Resultado |
|-------------|-----------|
| v1 (rollout puro) vs Aleatorio | 99/100 (99%) |
| v2 (rollout heurístico) vs Aleatorio | 100/100 (100%) |
| v2 vs v1 cara a cara | 71/100 (71%) a favor de v2 |
| 200 iteraciones vs Aleatorio | 40/40 (100%) |
| Como Rojo vs Aleatorio (120 iters) | 19/20 (95%) |
| Como Amarillo vs Aleatorio (120 iters) | 19/20 (95%) |
| MCTS vs sí mismo | ~50% (comportamiento simétrico esperado) |

---

## Enlace al repositorio

```
https://github.com/<grupo>/<repo>/blob/<tu-nombre>/<tu-nombre>/policy.py
```

> Reemplazar `<grupo>`, `<repo>` y `<tu-nombre>` con los valores reales del grupo.
