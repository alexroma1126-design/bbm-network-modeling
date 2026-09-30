# Tesis de Maestría en Ciencia de Datos

**Universidad Yachay Tech**
Escuela de Ciencias Matemáticas y Computacionales
Maestría en Ciencia de Datos

**Autor:** Héctor Alexander Román Potosí

## Título de trabajo

**Controlabilidad y asignación estratégica de actuadores en sistemas tipo BBM mediante teoría espectral de grafos y equilibrio de Nash**

---

# Resumen

Este repositorio contiene los componentes matemáticos, computacionales y de Ciencia de Datos de una investigación de maestría basada en sistemas de Benjamin--Bona--Mahony (BBM).

Una parte importante del proyecto estudia **redes de sistemas BBM no lineales**, donde:

- cada nodo representa un sistema BBM;
- las aristas representan acoplamiento entre sistemas;
- pueden introducirse actuadores localizados en determinados nodos;
- la dinámica se simula numéricamente mediante una aproximación espectral de Fourier y Runge--Kutta de cuarto orden.

La investigación no se limita a estudiar una sola simulación.

Se construyó un experimento computacional reproducible de:

$$
\boxed{1200\text{ simulaciones}}
$$

que posteriormente se transformó en un conjunto de datos estructurado para realizar:

- análisis exploratorio de datos;
- comparaciones pareadas;
- análisis de interacciones;
- visualización estática y dinámica;
- regresión lineal;
- Random Forest;
- validación agrupada para evitar fuga de información.

Por tanto, el modelo BBM funciona como **mecanismo generador de datos**, mientras que la Ciencia de Datos permite estudiar los patrones producidos por una población de simulaciones.

<p align="center">
  <img src="figures/network_trajectory/bbm_actuator_comparison.gif"
       alt="Comparación de ubicación de actuadores BBM"
       width="900">
</p>

---

# Pregunta de investigación computacional

La pregunta principal del estudio de redes es:

> **¿Cómo influyen la topología de la red, el acoplamiento, las condiciones iniciales y la ubicación de los actuadores en la dinámica no lineal y en las métricas de sincronización de una red de sistemas BBM?**

La idea central puede resumirse como:

$$
\text{modelo BBM}
\rightarrow
\text{red}
\rightarrow
\text{simulaciones}
\rightarrow
\text{datos}
\rightarrow
\text{patrones}
\rightarrow
\text{predicción}.
$$

---

# ¿Por qué este proyecto pertenece a Ciencia de Datos?

La ecuación BBM proporciona el modelo matemático que genera las observaciones.

La parte de Ciencia de Datos comienza cuando se diseña sistemáticamente el experimento y se analizan las simulaciones como datos.

En lugar de estudiar únicamente una trayectoria, se estudia una **población de experimentos computacionales**.

El flujo completo es:

$$
\boxed{
\text{BBM}
\rightarrow
1200\text{ experimentos}
\rightarrow
\text{dataset}
\rightarrow
\text{EDA}
\rightarrow
\text{Machine Learning}
}
$$

El proyecto incluye:

1. diseño experimental reproducible;
2. generación de datos sintéticos;
3. validación de integridad;
4. construcción de variables;
5. análisis exploratorio;
6. análisis de interacciones;
7. visualización;
8. modelado supervisado;
9. evaluación y validación de generalización.

Los datos son **sintéticos y generados por el modelo**. No se presentan como mediciones reales de una infraestructura física.

---

# Modelo matemático

Para el nodo \(i\), se considera una dinámica BBM de la forma

$$
(1-\partial_{xx})u_{i,t}
+
u_{i,x}
+
u_i u_{i,x}
=
\gamma
\sum_j w_{ij}(u_j-u_i)
+
\text{control}_i.
$$

Aquí:

- \(u_i(x,t)\) representa el estado BBM del nodo \(i\);
- \(w_{ij}\) representa la conexión entre los nodos \(i\) y \(j\);
- \(\gamma\) controla la intensidad del acoplamiento;
- el término de control representa una intervención externa localizada.

El término

$$
\gamma\sum_j w_{ij}(u_j-u_i)
$$

es un acoplamiento difusivo.

Si dos nodos tienen exactamente el mismo estado, su diferencia es cero y el acoplamiento entre ellos no produce corrección.

---

# Aproximación espectral

Cada estado BBM se aproxima mediante una expansión de Fourier:

$$
u_i^K(x,t)
=
\sum_{k=-K}^{K}
a_{i,k}(t)e^{ikx}.
$$

De esta forma, la ecuación en derivadas parciales se transforma en un sistema finito de ecuaciones diferenciales para los coeficientes

$$
a_{i,k}(t).
$$

La integración temporal se realiza mediante Runge--Kutta clásico de cuarto orden.

En los experimentos principales se utiliza:

$$
K=4,
\qquad
T=1,
\qquad
h=0.0025.
$$

---

# Grafos estudiados

Se consideran tres familias de grafos:

### Path

Una estructura tipo cadena:

```text
0 -- 1 -- 2 -- 3 -- ...
```

### Cycle

Los nodos forman un ciclo cerrado.

### Star

Un nodo central o hub se conecta con todos los demás nodos.

Estas estructuras permiten estudiar si la conectividad de la red modifica la dinámica de los sistemas BBM.

---

# Diseño experimental

El dataset principal utiliza:

$$
3\times5\times4\times5\times4
=
1200
$$

simulaciones.

| Factor | Valores |
|---|---|
| Topología | path, cycle, star |
| Número de nodos | 4, 5, 6, 7, 8 |
| Acoplamiento \(\gamma\) | 0, 0.25, 0.5, 1 |
| Condición inicial | 5 configuraciones |
| Escenario de actuación | 4 estrategias |

Los escenarios son:

```text
uncontrolled
single_0
single_middle
two_0_last
```

Por tanto:

$$
3
\times
5
\times
4
\times
5
\times
4
=
\boxed{1200}.
$$

Cada fila del dataset representa una simulación completa.

---

# Dataset

El conjunto principal se encuentra en:

[`data/generated/bbm_network_eda_v1.csv`](data/generated/bbm_network_eda_v1.csv)

Contiene:

$$
\boxed{1200\text{ filas}\times29\text{ variables}}.
$$

No contiene filas duplicadas.

Entre sus variables se encuentran características de:

- topología;
- tamaño de red;
- número de aristas;
- densidad;
- grados;
- condición inicial;
- acoplamiento;
- estrategia del actuador;
- esfuerzo de control;
- error de sincronización;
- normas de coeficientes;
- distancia respecto de la trayectoria no controlada.

---

# Error de sincronización

Una de las variables principales del trabajo es el error de sincronización de la red.

Conceptualmente mide cuánto difieren los estados de nodos conectados.

Un valor menor significa que los estados conectados son más similares.

Para estudiar el efecto de los actuadores se utiliza:

$$
\Delta E_{\mathrm{sync}}
=
E_{\mathrm{sync}}^{\mathrm{controlado}}(T)
-
E_{\mathrm{sync}}^{\mathrm{sin\ control}}(T).
$$

Por tanto:

$$
\Delta E_{\mathrm{sync}}<0
$$

indica que el experimento controlado terminó con menor error de sincronización que su baseline.

Mientras que:

$$
\Delta E_{\mathrm{sync}}>0
$$

indica que el error final aumentó.

Esta métrica describe sincronización.

**No constituye una prueba de controlabilidad no lineal.**

---

# Análisis exploratorio de datos

Las 900 simulaciones controladas se emparejaron con sus correspondientes 300 escenarios no controlados.

Esto permite comparar cada intervención con su baseline bajo exactamente:

- la misma topología;
- el mismo número de nodos;
- el mismo \(\gamma\);
- la misma condición inicial;
- el mismo esquema numérico.

---

## Efecto según tamaño de red

![Efecto según tamaño de red](figures/eda/01_delta_sync_vs_network_size.png)

El efecto observado de los actuadores cambia con el tamaño de la red y con la estrategia utilizada.

No existe una respuesta única que pueda explicarse solamente mediante el número de nodos.

---

## Interacción con la condición inicial

![Interacción condición inicial](figures/eda/04_initial_condition_interaction.png)

Este es uno de los resultados más importantes del EDA.

La misma estrategia de actuación puede:

- disminuir el error de sincronización bajo ciertas condiciones iniciales;
- aumentarlo bajo otras.

Por tanto, el efecto del actuador depende fuertemente del estado inicial del sistema.

---

## Interacción entre topología y estrategia

![Interacción topología y estrategia](figures/eda/05_topology_strategy_interaction.png)

La ubicación del actuador también interactúa con la estructura de la red.

Sin embargo, el grado del nodo por sí solo no explica completamente los resultados.

Por ejemplo, en una red cycle todos los nodos tienen el mismo grado, pero actuar sobre diferentes índices puede producir respuestas distintas debido a la distribución de las condiciones iniciales.

---

# Modelado predictivo

Después del análisis exploratorio se formuló un problema de regresión supervisada.

La variable objetivo es:

$$
\boxed{
y=\Delta E_{\mathrm{sync}}
}
$$

y se utilizan como predictores:

```text
graph_type
n_nodes
gamma
initial_condition_id
scenario
```

No se utilizan como predictores variables calculadas después de la simulación, como:

- `final_sync_error`;
- `sync_ratio`;
- `distance_from_uncontrolled`;
- normas finales.

Esto evita introducir información futura en el modelo.

---

# Dataset supervisado

De las 1200 simulaciones originales:

- 300 corresponden a escenarios sin control;
- 900 corresponden a escenarios controlados.

Por tanto, el dataset supervisado contiene:

$$
\boxed{900\text{ observaciones}}
$$

organizadas en:

$$
\boxed{300\text{ grupos experimentales}}.
$$

Cada grupo contiene:

```text
single_0
single_middle
two_0_last
```

asociados al mismo baseline no controlado.

---

# Prevención de fuga de información

Los tres escenarios controlados de un mismo experimento están relacionados.

Por ello no se utiliza una separación aleatoria simple de filas.

Se define un identificador:

```text
modeling_group_id
```

y se garantiza que:

$$
G_{\mathrm{train}}
\cap
G_{\mathrm{test}}
=
\varnothing.
$$

Los tres escenarios asociados a un baseline pertenecen completamente a entrenamiento o completamente a prueba.

Esto evita **data leakage**.

---

# Modelos comparados

Se evaluaron tres modelos:

### Predictor de la media

Funciona como baseline.

### Regresión lineal

Permite medir cuánto puede explicarse mediante una relación aproximadamente lineal.

### Random Forest Regressor

Permite capturar:

- relaciones no lineales;
- interacciones;
- efectos condicionales entre variables.

---

# Resultados del Machine Learning

Para evaluar estabilidad se realizaron **10 particiones agrupadas diferentes**.

Los resultados promedio fueron:

| Modelo | MAE | RMSE | \(R^2\) |
|---|---:|---:|---:|
| Baseline de la media | 0.03301 | 0.04365 | -0.0107 |
| Regresión lineal | 0.02745 | 0.03802 | 0.2287 |
| Random Forest | **0.00695** | **0.01009** | **0.9450** |

Para el Random Forest:

$$
\boxed{
R^2=0.9450\pm0.0106
}
$$

y

$$
\boxed{
RMSE=0.01009\pm0.00082.
}
$$

El Random Forest obtuvo menor RMSE que:

- la regresión lineal en **10 de 10** particiones;
- el baseline en **10 de 10** particiones.

![Rendimiento de modelos](figures/modeling/01_model_performance.png)

La diferencia entre la regresión lineal y Random Forest indica que existen relaciones no lineales e interacciones importantes en los datos.

---

# Predicción vs observación

![Predicción Random Forest](figures/modeling/02_rf_predicted_vs_actual.png)

Los valores predichos por Random Forest siguen de cerca la diagonal

$$
y=x,
$$

para la mayoría de los experimentos evaluados.

Esto indica buena capacidad predictiva **dentro del dominio experimental conocido**.

---

# Importancia de variables

Se utilizó permutation importance sobre datos de prueba.

![Importancia de variables](figures/modeling/03_rf_permutation_importance.png)

El orden observado fue aproximadamente:

1. estrategia del actuador;
2. condición inicial;
3. tamaño de la red;
4. acoplamiento \(\gamma\);
5. topología.

Los dos factores con mayor información predictiva fueron:

$$
\boxed{\text{estrategia del actuador}}
$$

y

$$
\boxed{\text{condición inicial}}.
$$

Esto coincide con los patrones observados previamente durante el EDA.

La importancia predictiva no debe interpretarse automáticamente como causalidad.

---

# ¿Qué significa realmente el \(R^2=0.945\)?

Se realizaron dos tipos distintos de validación.

## Caso 1: niveles conocidos

El modelo recibe experimentos nuevos, pero todos los niveles de:

- topología;
- tamaño;
- \(\gamma\);
- condición inicial;
- estrategia;

ya estuvieron representados durante entrenamiento.

En esta situación:

$$
R^2_{\mathrm{RF}}
\approx
0.95.
$$

---

## Caso 2: condición inicial completamente nueva

Se realizó además una validación donde una condición inicial completa queda fuera del entrenamiento.

Los valores de \(R^2\) del Random Forest fueron:

```text
 0.294
-0.060
-2.963
 0.109
 0.409
```

![Regímenes de validación](figures/modeling/04_validation_regimes_v2.png)

El rendimiento disminuye fuertemente.

La razón es importante:

`initial_condition_id` es solamente una **categoría**.

Por ejemplo:

```text
initial_condition_id = 2
```

no contiene por sí mismo información matemática que describa cómo es esa condición inicial.

Por tanto, el modelo puede aprender muy bien las condiciones iniciales observadas, pero no puede extrapolar automáticamente hacia una condición inicial desconocida.

Una futura mejora consistiría en representar las condiciones iniciales mediante características físicas o espectrales reales.

---

# Visualización dinámica de las redes

Además del análisis tabular, se creó un pipeline para exportar trayectorias completas desde C++ y reconstruir los campos BBM en Python.

La cadena es:

$$
\text{RK4}
\rightarrow
a_{i,k}(t)
\rightarrow
\text{CSV}
\rightarrow
\text{reconstrucción Fourier}
\rightarrow
u_i(x,t).
$$

Cada frame de las animaciones corresponde a un snapshot realmente exportado.

No se interpolan estados artificiales.

---

# Evolución de una red Path

![Animación Path](figures/network_trajectory/bbm_path_animation.gif)

El piloto utiliza tres nodos y

$$
\gamma=0.5.
$$

En esta trayectoria particular:

$$
E_{\mathrm{sync}}(0)
=
0.67082,
$$

$$
E_{\mathrm{sync}}(0.5)
=
0.47370,
$$

$$
E_{\mathrm{sync}}(1)
=
0.33825.
$$

Visualmente se observa cómo las trayectorias de los nodos se aproximan durante esta simulación.

Este resultado corresponde a un experimento concreto y no constituye un teorema general de sincronización.

---

# Comparación de topologías

Para comparar estructuras genuinamente diferentes se utilizaron cinco nodos:

$$
P_5,
\qquad
C_5,
\qquad
K_{1,4}.
$$

![Comparación de topologías](figures/network_trajectory/bbm_topology_comparison.gif)

Se mantienen iguales:

- condición inicial;
- \(K\);
- \(T\);
- \(h\);
- \(\gamma\).

La diferencia es la estructura de conexiones del grafo.

No se utiliza el error de sincronización para afirmar que una topología sea universalmente superior a otra, porque la métrica depende también del conjunto de aristas sobre el que se calcula.

---

# Comparación de ubicación del actuador

Se estudió además una red path de cinco nodos:

```text
0 -- 1 -- 2 -- 3 -- 4
```

con tres escenarios:

```text
sin control
actuador en nodo 0
actuador en nodo 2
```

![Comparación de actuadores](figures/network_trajectory/bbm_actuator_comparison.gif)

Los dos experimentos controlados usan el mismo perfil espacial:

$$
b(x)=\cos x,
$$

la misma amplitud:

$$
A=0.6,
$$

y exactamente el mismo esfuerzo cuadrático:

$$
A^2T
=
0.36.
$$

En este experimento particular, para \(t=1\):

| Escenario | \(E_{\mathrm{sync}}(1)\) |
|---|---:|
| Sin control | 0.11095 |
| Actuador en nodo 0 | 0.10547 |
| Actuador en nodo 2 | 0.15338 |

Esto muestra que **la ubicación del actuador puede modificar considerablemente la respuesta dinámica incluso cuando el esfuerzo de control es idéntico**.

No demuestra que un nodo sea universalmente más controlable que otro.

---

# Validación computacional

La infraestructura incluye validaciones para:

- RHS de la red;
- integración RK4;
- actuadores localizados;
- integración controlada;
- observables de red;
- exportación determinista de trayectorias;
- reconstrucción de Fourier;
- simetría conjugada;
- valores numéricos finitos;
- recomputación independiente de \(E_{\mathrm{sync}}\);
- igualdad de condiciones iniciales entre experimentos comparados;
- ausencia de grupos compartidos entre entrenamiento y prueba.

Actualmente:

$$
\boxed{6/6}
$$

pruebas C++ de red pasan correctamente.

Además:

$$
\boxed{12/12}
$$

pruebas Python de trayectorias y reconstrucción pasan correctamente.

---

# Reproducibilidad

## Dependencias Python

Instalar:

```powershell
python -m pip install -r requirements.txt
```

El entorno utilizado durante el desarrollo actual fue:

```text
Python          3.13.14
NumPy           2.4.6
pandas          3.0.3
matplotlib      3.11.2
Pillow          12.3.0
scikit-learn    1.8.0
```

Estas son versiones verificadas del entorno utilizado, no necesariamente versiones mínimas requeridas.

---

# Compilar la red BBM

El proyecto de red utiliza:

- C++20;
- CMake >= 3.20;
- Ninja en el entorno de desarrollo actual.

Configurar:

```powershell
cmake -S code\bbm_network `
      -B build\bbm_network_visualization `
      -G Ninja `
      -DCMAKE_BUILD_TYPE=Release
```

Compilar:

```powershell
cmake --build build\bbm_network_visualization
```

Ejecutar validaciones:

```powershell
ctest --test-dir build\bbm_network_visualization --output-on-failure
```

---

# Regenerar el dataset de 1200 simulaciones

El generador C++ escribe el CSV mediante la salida estándar.

```powershell
New-Item -ItemType Directory -Force data\generated | Out-Null

& .\build\bbm_network_visualization\bbm_network_generate_dataset.exe |
    Set-Content data\generated\bbm_network_eda_v1.csv -Encoding UTF8
```

---

# Ejecutar el EDA

```powershell
python -B code\eda\analyze_bbm_network.py
```

---

# Construir el dataset supervisado

```powershell
python -B code\eda\build_bbm_network_modeling_dataset.py
```

---

# Entrenar los modelos

```powershell
python -B code\eda\model_bbm_network.py
```

---

# Validación repetida

```powershell
python -B code\eda\validate_bbm_network_models_repeated.py
```

---

# Validación con condición inicial no observada

```powershell
python -B code\eda\validate_bbm_network_models.py
```

---

# Regenerar figuras de Machine Learning

```powershell
python -B code\eda\visualize_bbm_network_models.py
```

---

# Validar reconstrucción Fourier

```powershell
python -B code\eda\test_bbm_network_trajectory.py
```

---

# Estructura relevante del repositorio

```text
Tesis_Maestria/
│
├── chapters/
│   └── capítulos de la tesis
│
├── code/
│   ├── bbm_nonlinear/
│   │   └── BBM espectral y control
│   │
│   ├── bbm_network/
│   │   └── redes, grafos, actuadores y generación del dataset
│   │
│   └── eda/
│       └── EDA, Machine Learning, validación y visualización
│
├── data/
│   └── generated/
│       └── bbm_network_eda_v1.csv
│
├── figures/
│   ├── eda/
│   ├── modeling/
│   └── network_trajectory/
│
├── experiments/
│
├── docs/
│
├── requirements.txt
│
└── README.md
```

Los directorios de compilación, trayectorias temporales y datasets derivados permanecen fuera del control de versiones.

---

# Relación con la tesis completa

Este estudio computacional se integra dentro de una investigación más amplia.

La tesis también estudia:

- ecuación BBM;
- análisis espectral;
- aproximaciones de Fourier;
- controlabilidad;
- criterio PBH;
- Gramianos;
- perfiles de actuadores;
- teoría espectral de grafos;
- asignación estratégica de actuadores;
- mejores respuestas;
- equilibrio de Nash;
- bienestar;
- precio de la anarquía.

La parte de Ciencia de Datos aporta una capa empírica complementaria:

$$
\boxed{
\text{matemática}
+
\text{simulación}
+
\text{datos}
+
\text{aprendizaje automático}
}
$$

---

# Limitaciones

Los resultados deben interpretarse dentro del alcance del experimento.

**Datos sintéticos.**
Las observaciones son generadas por el modelo BBM y no corresponden a mediciones de una infraestructura física real.

**Sin calibración física específica.**
No se afirma que la red modele directamente una ciudad, sistema hidráulico, red eléctrica u otra infraestructura concreta.

**Sincronización no significa controlabilidad.**
Reducir \(E_{\mathrm{sync}}\) no demuestra controlabilidad no lineal.

**Distancia entre trayectorias tampoco significa controlabilidad.**
La distancia modal respecto del caso sin control únicamente cuantifica cuánto cambia la trayectoria.

**PBH y Gramiano tienen otro alcance.**
Los resultados PBH y de Gramianos utilizados en otras partes de la tesis corresponden a sistemas linealizados o truncados de dimensión finita.

**Random Forest interpola mejor que extrapola.**
El alto \(R^2\) corresponde al dominio factorial observado. La generalización hacia condiciones iniciales completamente nuevas es considerablemente más difícil.

**Las animaciones son ilustrativas.**
Una animación individual representa un experimento concreto. Las conclusiones generales deben apoyarse en el dataset completo.

---

# Próximas extensiones

Entre las extensiones naturales del trabajo se encuentran:

- representar las condiciones iniciales mediante características físicas y espectrales;
- estudiar tamaños de red no observados durante entrenamiento;
- estudiar nuevos valores de \(\gamma\);
- incorporar descriptores espectrales del grafo;
- estudiar nuevas familias de redes;
- relacionar cuidadosamente métricas lineales de controlabilidad con la respuesta no lineal observada;
- extender el análisis estratégico de actuadores a redes mayores.

---

# Compilación de la tesis

El manuscrito científico principal se desarrolla en inglés.

Para compilar:

```powershell
latexmk -pdf -interaction=nonstopmode -file-line-error main.tex
```

Para limpiar:

```powershell
latexmk -C
```

---

# Estado actual

Actualmente el repositorio contiene:

- infraestructura BBM no lineal validada;
- redes BBM acopladas;
- actuadores localizados;
- dataset reproducible de 1200 experimentos;
- análisis exploratorio de datos;
- comparaciones pareadas;
- reconstrucción Fourier validada;
- visualizaciones estáticas;
- tres animaciones científicas;
- regresión lineal;
- Random Forest;
- validación agrupada sin fuga de información;
- análisis de límites de generalización;
- componentes matemáticos de control, grafos y juegos estratégicos.

La investigación de tesis continúa activa y estos componentes podrán integrarse y ampliarse durante el desarrollo del manuscrito.
