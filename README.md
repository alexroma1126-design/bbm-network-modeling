# Análisis de redes BBM mediante Ciencia de Datos

**Proyecto académico — Análisis de Bases de Datos**
**Maestría en Ciencia de Datos — Universidad Yachay Tech**

**Autor:** Héctor Alexander Román Potosí

---

## Descripción del proyecto

Este proyecto estudia el comportamiento de redes de sistemas Benjamin–Bona–Mahony (BBM) mediante técnicas de análisis de datos y aprendizaje automático.

La idea principal es utilizar un modelo matemático como generador de datos sintéticos. Cada simulación corresponde a una configuración específica de red, acoplamiento, condición inicial y estrategia de actuación.

A partir de estas simulaciones se construyó un dataset estructurado de:

$$
\boxed{1200\text{ experimentos}}
$$

que posteriormente se utiliza para:

- análisis exploratorio de datos;
- comparación de escenarios controlados y no controlados;
- estudio de interacciones entre variables;
- visualización de patrones;
- regresión lineal;
- Random Forest;
- validación agrupada para evitar fuga de información.

La cadena conceptual del proyecto es:

$$
\boxed{
\text{modelo BBM}
\rightarrow
\text{red}
\rightarrow
\text{simulaciones}
\rightarrow
\text{dataset}
\rightarrow
\text{EDA}
\rightarrow
\text{Machine Learning}
}
$$

Los datos utilizados son sintéticos y se generan completamente mediante simulación numérica.

---

## Pregunta de análisis

La pregunta principal del proyecto es:

> **¿Cómo influyen la topología de la red, el acoplamiento, las condiciones iniciales y la ubicación de los actuadores en la dinámica y en las métricas de sincronización de una red de sistemas BBM?**

El objetivo no es analizar una única trayectoria, sino estudiar una población de experimentos computacionales y buscar patrones entre configuraciones distintas.

---

## ¿Por qué este proyecto pertenece a Ciencia de Datos?

La ecuación BBM proporciona el mecanismo matemático que genera las observaciones.

La parte de Ciencia de Datos comienza cuando esas simulaciones se organizan como un conjunto estructurado de datos y se estudian de forma sistemática.

En lugar de observar solamente una trayectoria:

$$
u(x,t)
$$

se analizan cientos de configuraciones con diferentes características experimentales.

El flujo general es:

$$
\text{modelo matemático}
\rightarrow
\text{diseño experimental}
\rightarrow
\text{simulación}
\rightarrow
\text{dataset}
\rightarrow
\text{análisis}
\rightarrow
\text{predicción}
$$

Cada fila del dataset representa una simulación completa.

---

## Estructura general del trabajo

El proyecto se divide en cuatro etapas principales:

1. **Modelado matemático:** definición de la dinámica BBM y del acoplamiento mediante grafos.
2. **Generación de datos:** ejecución reproducible de múltiples simulaciones.
3. **Análisis exploratorio:** estudio de patrones, interacciones y efectos de los actuadores.
4. **Modelado predictivo:** comparación entre modelos supervisados y evaluación de su capacidad de generalización.

---

# Fundamentos matemáticos


## 1. Estado de un sistema BBM

Cada nodo de la red representa un sistema Benjamin–Bona–Mahony (BBM).

Su estado se escribe como

$$
u_i(x,t),
$$

donde:

- $i$ identifica el nodo de la red;
- $x$ representa la posición espacial;
- $t$ representa el tiempo.

Para un instante fijo $t=t_0$,

$$
u_i(x,t_0)
$$

representa un perfil espacial completo.

Por tanto, el estado de un nodo no es simplemente un número. Es una función espacial que evoluciona con el tiempo.

Si además fijamos una posición $x=x_0$, entonces

$$
u_i(x_0,t)
$$

sí puede interpretarse como una señal escalar que cambia con el tiempo.

Esta distinción será importante cuando utilicemos Fourier para representar numéricamente cada estado BBM.

La siguiente figura ilustra cómo una solución $u(x,t)$ puede interpretarse simultáneamente como una superficie espacio--tiempo y como una familia de perfiles espaciales obtenidos al fijar distintos instantes.

<p align="center">
  <img src="figures/mathematical_background/bbm_solution_space_time.png" alt="Interpretación espacio-temporal de una solución BBM" width="850">
</p>

*Figura de contexto matemático: representación espacio--temporal de una solución BBM.*


---

## 2. Ecuación BBM en un nodo

Una forma esquemática de la dinámica utilizada para el nodo $i$ es

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
\mathrm{control}_i.
$$

En esta expresión:

- $u_i(x,t)$ es el estado del nodo $i$;
- $u_{i,t}$ representa la evolución temporal;
- $u_{i,x}$ representa la variación espacial;
- $u_i u_{i,x}$ introduce la no linealidad;
- $\gamma$ controla la intensidad del acoplamiento;
- $w_{ij}$ describe las conexiones de la red;
- $\mathrm{control}_i$ representa una intervención externa.

La ecuación combina tres elementos:

$$
\boxed{
\text{dinámica BBM}
+
\text{interacción con la red}
+
\text{control}
}
$$

---

## 3. De un sistema BBM a una red

En lugar de estudiar una única función

$$
u(x,t),
$$

se consideran varios sistemas:

$$
u_0(x,t),
\quad
u_1(x,t),
\quad
\dots,
\quad
u_{n-1}(x,t).
$$

Cada función corresponde a un nodo del grafo.

Por ejemplo, una red tipo path con tres nodos es

```text
0 --- 1 --- 2
```

y representa tres sistemas BBM conectados:

```text
BBM_0 --- BBM_1 --- BBM_2
```

Las aristas indican qué sistemas pueden interactuar directamente.

En este ejemplo:

- el nodo 0 interactúa con el nodo 1;
- el nodo 1 interactúa con los nodos 0 y 2;
- el nodo 2 interactúa con el nodo 1.

La teoría de grafos permite representar esta estructura de interacción de forma matricial.

---

## 4. Matriz de adyacencia

La estructura del grafo puede representarse mediante una matriz de adyacencia $A$.

Para el grafo

```text
0 --- 1 --- 2
```

se tiene

$$
A=
\begin{pmatrix}
0 & 1 & 0\\
1 & 0 & 1\\
0 & 1 & 0
\end{pmatrix}.
$$

El elemento $a_{ij}$ indica si dos nodos están conectados.

En un grafo no dirigido y sin pesos,

$$
a_{ij}=
\begin{cases}
1, & \text{si } i \text{ y } j \text{ están conectados},\\
0, & \text{en otro caso.}
\end{cases}
$$

---

## 5. Matriz de grados

El grado de un nodo es el número de conexiones que posee.

Para el mismo grafo,


$$
d_0=1,
\quad
d_1=2,
\quad
d_2=1.
$$

La matriz de grados es diagonal:

$$
D=
\begin{pmatrix}
1 & 0 & 0\\
0 & 2 & 0\\
0 & 0 & 1
\end{pmatrix}.
$$

En general,

$$
d_i=\sum_j a_{ij}.
$$

Por tanto, la diagonal de $D$ contiene exactamente el número de vecinos de cada nodo.

---

## 6. Laplaciano del grafo

El Laplaciano se define como

$$
\boxed{L=D-A}.
$$

Para el ejemplo anterior,

$$
L=
\begin{pmatrix}
1 & -1 & 0\\
-1 & 2 & -1\\
0 & -1 & 1
\end{pmatrix}.
$$

La diagonal de $L$ contiene los grados de los nodos.

Fuera de la diagonal:

- aparece $-1$ si existe una arista entre dos nodos;
- aparece $0$ si no existe conexión directa.

Cada fila del Laplaciano suma cero.

Por tanto,

$$
L\mathbf{1}=0.
$$

Si todos los nodos tienen el mismo estado, el Laplaciano no detecta diferencias entre ellos.

---

## 7. Interpretación del Laplaciano

Sea

$$
U=
\begin{pmatrix}
u_0\\
u_1\\
u_2
\end{pmatrix}.
$$

Entonces,

$$
LU=
\begin{pmatrix}
u_0-u_1\\
-u_0+2u_1-u_2\\
u_2-u_1
\end{pmatrix}.
$$

El Laplaciano compara el estado de cada nodo con los estados de sus vecinos.

Como el acoplamiento utiliza la forma

$$
\sum_j w_{ij}(u_j-u_i),
$$

este término equivale, con la convención utilizada, a

$$
-(LU)_i.
$$

Por tanto, el acoplamiento global puede escribirse como

$$
\boxed{-\gamma LU}.
$$

Esta expresión resume en una única operación matricial todas las diferencias entre nodos conectados.

---

## 8. Interpretación espectral

Los autovalores y autovectores del Laplaciano satisfacen

$$
Lv=\lambda v.
$$

Los autovectores pueden interpretarse como patrones de variación sobre la red, mientras que los autovalores indican qué tan intensamente el Laplaciano detecta esos patrones.

El autovalor

$$
\lambda_1=0
$$

está asociado al modo constante, es decir, al caso en que todos los nodos tienen el mismo estado.

El segundo autovalor,

$$
\lambda_2,
$$

se conoce como conectividad algebraica.

Para un grafo conectado,

$$
\lambda_2>0.
$$

Esta cantidad contiene información sobre la conectividad global de la red.

---

## 9. Aproximación espectral de Fourier

Cada estado BBM se aproxima mediante una expansión de Fourier:

$$
u_i^K(x,t)
=
\sum_{k=-K}^{K}
a_{i,k}(t)e^{ikx}.
$$

La función espacial $u_i(x,t)$ queda representada por un conjunto finito de coeficientes

$$
a_{i,k}(t).
$$

Con esta aproximación, el problema continuo se transforma en un sistema finito de ecuaciones diferenciales ordinarias para los coeficientes de Fourier.

En los experimentos principales se utiliza:

$$
K=4,
\quad
T=1,
\quad
h=0.0025.
$$


---

## 10. Integración temporal

La evolución temporal de los coeficientes de Fourier se calcula mediante Runge–Kutta clásico de cuarto orden.

El flujo numérico puede resumirse como

$$
u_i(x,0)
\rightarrow
a_{i,k}(0)
\rightarrow
\text{RK4}
\rightarrow
a_{i,k}(t)
\rightarrow
u_i(x,t).
$$

Cada paso temporal produce un nuevo estado de la red y permite reconstruir las trayectorias de los sistemas BBM.


---

## 11. Del modelo matemático al dataset

La cadena completa de generación de datos es

$$
\boxed{
\text{ecuaciones BBM}
\rightarrow
\text{grafo}
\rightarrow
\text{Fourier}
\rightarrow
\text{RK4}
\rightarrow
\text{simulación}
\rightarrow
\text{variables}
\rightarrow
\text{dataset}
}
$$

Cada configuración experimental produce una simulación completa.

A partir de esa simulación se extraen variables relacionadas con:

- topología;
- tamaño de red;
- acoplamiento;
- condición inicial;
- estrategia del actuador;
- error de sincronización;
- esfuerzo de control;
- normas de coeficientes;
- diferencias respecto del escenario no controlado.

Por tanto, el modelo matemático funciona como mecanismo generador de datos sintéticos que luego se analizan desde Ciencia de Datos.

---

# Diseño experimental

El dataset principal se construye a partir de un diseño factorial que combina varios factores experimentales.

La combinación es

$$
3\times5\times4\times5\times4
=
1200.
$$

Por tanto, se generan

$$
\boxed{1200\text{ simulaciones}}.
$$

Cada simulación corresponde a una configuración única de los siguientes factores:

| Factor | Valores |
|---|---|
| Topología | path, cycle, star |
| Número de nodos | 4, 5, 6, 7, 8 |
| Acoplamiento $\gamma$ | 0, 0.25, 0.5, 1 |
| Condición inicial | 5 configuraciones |
| Escenario de actuación | 4 estrategias |

---

## Topologías

Se estudian tres familias de grafos:

- **Path:** los nodos se conectan formando una cadena.
- **Cycle:** los nodos forman un ciclo cerrado.
- **Star:** un nodo central o hub se conecta con todos los demás.

Estas topologías permiten evaluar cómo la estructura de la red modifica la dinámica de los sistemas BBM.

---

## Tamaño de red

Se consideran redes con

$$
n=4,5,6,7,8.
$$

Esto permite estudiar si el efecto de los actuadores y el acoplamiento dependen del tamaño de la red.

---

## Acoplamiento de la red

El parámetro

$$
\gamma
$$

controla la intensidad de interacción entre nodos conectados.

En el experimento se utilizan los valores

$$
\gamma \in \{0,\;0.25,\;0.5,\;1\}.
$$

Cuando

$$
\gamma=0,
$$

el acoplamiento del grafo no influye en la dinámica.

Al aumentar $\gamma$, la interacción entre los nodos se hace más intensa.

---

## Condiciones iniciales

Cada experimento se ejecuta bajo una de cinco configuraciones iniciales.

La condición inicial es importante porque determina el estado de partida de cada red BBM.

El análisis exploratorio mostrará posteriormente que la respuesta a un mismo actuador puede cambiar según la condición inicial.

---

## Escenarios de actuación

Se consideran cuatro escenarios:

```text
uncontrolled
single_0
single_middle
two_0_last
```

Su interpretación es:

- `uncontrolled`: no se aplica actuador;
- `single_0`: se actúa sobre el nodo 0;
- `single_middle`: se actúa sobre un nodo central;
- `two_0_last`: se utilizan dos actuadores, en los extremos de la red.

---

## Una fila del dataset

Cada fila del dataset representa una simulación completa.

En consecuencia, una observación no representa un nodo aislado ni un único instante de tiempo, sino el resumen de un experimento completo.

Esto es importante para interpretar correctamente las variables del dataset.

---

# Métrica de sincronización

Para comparar el comportamiento de los nodos de una red se utiliza una métrica de error de sincronización.

De forma conceptual, esta métrica mide qué tan diferentes son los estados de los nodos en un instante dado.

Cuando el error es pequeño, los nodos se encuentran más cerca de un comportamiento similar.

En cambio, un valor mayor indica mayor diferencia entre los estados de la red.

---

## Comparación respecto del escenario sin control

Para cuantificar el efecto de una estrategia de actuación, se utiliza la diferencia

$$
\Delta E_{\text{sync}}
=
E_{\text{sync}}^{\text{controlado}}(T)
-
E_{\text{sync}}^{\text{sin control}}(T).
$$

Por tanto:

- si $\Delta E_{\text{sync}}<0$, la estrategia controlada termina con un error de sincronización menor que el escenario sin control;
- si $\Delta E_{\text{sync}}=0$, no hay diferencia final respecto del baseline;
- si $\Delta E_{\text{sync}}>0$, el error final es mayor que en el escenario sin control.

Esta convención es fundamental para interpretar las figuras, tablas y modelos predictivos del proyecto.

---

## Interpretación correcta de $\Delta E_{\text{sync}}$

Un valor negativo no debe interpretarse como una demostración de controlabilidad.

Significa únicamente que, para esa configuración experimental concreta, el error final de sincronización fue menor con la estrategia de actuación empleada.

De forma análoga, un valor positivo no implica que el actuador sea inútil de forma general, sino que en esa simulación concreta el error final fue mayor que en el baseline.

---

# Visualización dinámica de las simulaciones

Las trayectorias completas contienen mucha más información que una única métrica final. Por esta razón, además del dataset tabular, el proyecto incluye animaciones construidas a partir de los estados simulados de la red.

Estas visualizaciones no sustituyen al análisis cuantitativo. Su función es complementar las métricas y permitir observar cómo evoluciona espacial y temporalmente cada sistema BBM.

---

## 1. Evolución de una red tipo path

<p align="center">
  <img src="figures/network_trajectory/bbm_path_animation.gif" alt="Evolución temporal de una red BBM tipo path" width="850">
</p>

*Animación de una red BBM tipo path reconstruida a partir de los estados obtenidos durante la simulación.*

Esta animación muestra la evolución temporal de los perfiles espaciales asociados a los nodos de una red tipo path.

Para cada instante $t$, cada curva representa un estado

$$
u_i(x,t).
$$

Al avanzar el tiempo pueden observarse:

- cambios en la forma de los perfiles;
- propagación de diferencias entre nodos conectados;
- aproximación o separación entre las trayectorias;
- efecto acumulado del acoplamiento de la red.

La animación se obtiene a partir de los estados numéricos generados durante la integración temporal y de la reconstrucción de Fourier. Por tanto, los cuadros representan estados calculados por la simulación y no una interpolación visual artificial.

Esta figura permite conectar directamente la ecuación BBM con los datos que posteriormente se resumen en el dataset.

---

## 2. Comparación entre topologías

<p align="center">
  <img src="figures/network_trajectory/bbm_topology_comparison.gif" alt="Comparación dinámica entre topologías BBM" width="850">
</p>

*Comparación dinámica entre redes path, cycle y star bajo una configuración experimental común.*

Aquí se comparan las tres topologías principales del estudio:

```text
path
cycle
star
```

La condición inicial, el horizonte temporal y los demás parámetros del experimento se mantienen comparables para aislar visualmente el efecto de la estructura de la red.

La topología modifica la matriz de adyacencia $A$, la matriz de grados $D$ y, en consecuencia, el Laplaciano

$$
L=D-A.
$$

Por ello, aunque los nodos individuales sigan la misma dinámica BBM, el patrón de interacción entre ellos cambia.

La animación sirve como evidencia visual de que la estructura del grafo puede modificar la evolución colectiva. Sin embargo, no debe interpretarse como una demostración de que una topología sea universalmente superior a otra: la respuesta también depende del tamaño de la red, del acoplamiento y de las condiciones iniciales.

---

## 3. Comparación de estrategias de actuación

<p align="center">
  <img src="figures/network_trajectory/bbm_actuator_comparison.gif" alt="Comparación de estrategias de actuación en una red BBM" width="850">
</p>

*Comparación de la dinámica de una misma red bajo distintas ubicaciones de actuadores.*

Esta animación compara escenarios de actuación manteniendo fija la estructura principal del experimento.

El objetivo visual es responder una pregunta sencilla:

> ¿Cambiar la ubicación del actuador modifica la respuesta de la red?

La comparación permite observar que dos estrategias con un esfuerzo de control comparable pueden producir trayectorias diferentes.

Esto motiva una de las variables centrales del dataset:

```text
scenario
```

y justifica estudiar de manera sistemática las estrategias

```text
uncontrolled
single_0
single_middle
two_0_last
```

La animación no demuestra por sí sola cuál estrategia es óptima. Su función es mostrar por qué la ubicación de los actuadores constituye un factor experimental relevante que posteriormente se analiza con estadística descriptiva y modelos supervisados.

---

## De las animaciones a las variables

Las animaciones muestran trayectorias completas, mientras que el dataset necesita representaciones tabulares que puedan compararse entre cientos de experimentos.

Por ello, cada simulación se resume mediante variables numéricas como el error de sincronización final, diferencias respecto del escenario sin control, esfuerzo de actuación y otras características derivadas.

El flujo conceptual es

$$
\boxed{
\text{trayectoria dinámica}
\rightarrow
\text{métricas}
\rightarrow
\text{fila del dataset}
}
$$

Esta transformación permite pasar de la observación cualitativa de una trayectoria a un análisis reproducible sobre las $1200$ simulaciones.

---

# Análisis exploratorio de datos

Una vez construidas las $1200$ simulaciones, el siguiente objetivo es estudiar cómo cambia el comportamiento de la red al modificar los factores experimentales.

El análisis exploratorio se centra especialmente en la variable

$$
\Delta E_{\text{sync}},
$$

porque permite comparar cada estrategia de actuación con su escenario de referencia sin control.

En esta sección se utilizan visualizaciones que permiten estudiar tres preguntas:

1. ¿Cómo cambia el efecto de la actuación con el tamaño de la red?
2. ¿Hasta qué punto la condición inicial modifica la respuesta?
3. ¿Existe interacción entre topología y estrategia de actuación?

---

## 1. Efecto del tamaño de la red

<p align="center">
  <img src="figures/eda/01_delta_sync_vs_network_size.png" alt="Delta de sincronización frente al tamaño de la red" width="850">
</p>

*Variación de $\Delta E_{\text{sync}}$ según el número de nodos y la estrategia de actuación.*

En esta figura, el eje horizontal representa el tamaño de la red y el eje vertical representa

$$
\Delta E_{\text{sync}}.
$$

La línea de referencia

$$
\Delta E_{\text{sync}}=0
$$

separa dos comportamientos:

- valores negativos indican un error final de sincronización menor que el baseline;
- valores positivos indican un error final mayor que el escenario sin control.

La figura permite observar que el efecto de una estrategia de actuación no es constante cuando cambia el número de nodos.

Por tanto, el tamaño de la red no debe tratarse únicamente como una característica descriptiva: puede interactuar con la ubicación y el número de actuadores.

Esta observación justifica incluir `n_nodes` como variable explicativa en la etapa de modelado supervisado.

---

## 2. Interacción con la condición inicial

<p align="center">
  <img src="figures/eda/04_initial_condition_interaction.png" alt="Interacción entre condición inicial y estrategia de actuación" width="850">
</p>

*Efecto de la estrategia de actuación bajo diferentes condiciones iniciales.*

Esta visualización estudia una cuestión especialmente importante: una misma estrategia puede producir respuestas distintas cuando cambia el estado inicial de la red.

Las cinco configuraciones iniciales no son simples etiquetas administrativas. Representan estados de partida diferentes para las simulaciones BBM.

La figura muestra que el efecto de la actuación depende de esa configuración inicial y que, por tanto, no resulta adecuado interpretar una estrategia de manera aislada.

Esto anticipa una dificultad que aparecerá posteriormente en la validación de los modelos: aprender patrones dentro de condiciones iniciales conocidas es más sencillo que extrapolar a una condición inicial completamente nueva.

Por esta razón, `initial_condition_id` se conserva como una variable explícita durante el modelado.

---

## 3. Interacción entre topología y estrategia

<p align="center">
  <img src="figures/eda/05_topology_strategy_interaction.png" alt="Interacción entre topología y estrategia de actuación" width="850">
</p>

*Comparación de $\Delta E_{\text{sync}}$ para distintas combinaciones de topología y estrategia.*

Esta figura permite estudiar conjuntamente dos factores:

$$
\text{topología}
\qquad\text{y}\qquad
\text{estrategia de actuación}.
$$

La pregunta ya no es solamente si una estrategia funciona mejor o peor en promedio, sino si su efecto cambia cuando cambia la estructura de conexiones de la red.

Esto es importante porque path, cycle y star poseen matrices de adyacencia y Laplacianos diferentes.

En consecuencia, una misma ubicación de actuadores puede interactuar de forma distinta con la estructura del grafo.

La figura debe interpretarse como evidencia de interacción experimental y no como una clasificación universal de topologías o estrategias.

---

## Lectura conjunta del EDA

Las tres visualizaciones conducen a una misma conclusión metodológica: la respuesta de la red BBM depende de combinaciones de factores y no únicamente de efectos individuales.

De forma esquemática,

$$
\boxed{
\Delta E_{\text{sync}}
=
f(
\text{topología},
\text{tamaño},
\gamma,
\text{condición inicial},
\text{estrategia}
)
}
$$

Esta relación no se plantea como una ecuación analítica exacta, sino como la pregunta predictiva que se estudiará mediante modelos supervisados.

El análisis exploratorio sugiere que pueden existir relaciones no lineales e interacciones entre variables. Por esta razón, en la siguiente etapa se compara un modelo lineal con un modelo Random Forest.

---

# Modelado supervisado

El objetivo de la etapa predictiva es estimar la variable

$$
\Delta E_{\text{sync}},
$$

a partir de características del diseño experimental.

Las variables de entrada utilizadas son:

```text
graph_type
initial_condition_id
scenario
n_nodes
gamma
```

Esta selección evita utilizar como predictores variables calculadas después de la simulación que podrían contener información directa sobre el resultado final.

---

## Prevención de fuga de información
Cada configuración base sin control genera tres escenarios controlados. Por tanto, esas tres observaciones no son independientes: comparten la misma topología, tamaño, condición inicial y acoplamiento.

Para evitar que información de un mismo experimento base aparezca simultáneamente en entrenamiento y prueba, se utiliza la variable

```text
modeling_group_id
```

como identificador de grupo.

La división se realiza mediante `GroupShuffleSplit`, con un 20 % de los grupos reservados para prueba.

el diseño utiliza:

- 240 grupos de entrenamiento;
- 60 grupos de prueba;
- 720 filas de entrenamiento;
- 180 filas de prueba;
- 0 grupos compartidos entre train y test.

De forma matemática,

$$
G_{\text{train}}\cap G_{\text{test}}=\varnothing.
$$

Esta restricción es necesaria para que la evaluación mida generalización sobre configuraciones no vistas durante el entrenamiento.

---

## Modelos comparados

### 1. Baseline de la media

El primer modelo utiliza como predicción el promedio de $\Delta E_{\text{sync}}$ en el conjunto de entrenamiento.

Este modelo no intenta aprender relaciones entre variables. Su función es servir como referencia mínima para los modelos más complejos.

---

### 2. Regresión lineal

La regresión lineal se utiliza como un modelo interpretable para evaluar si una combinación lineal de los factores experimentales puede explicar parte de la variación de $\Delta E_{\text{sync}}$.

Las variables categóricas se codifican mediante one-hot encoding y las variables numéricas se estandarizan.

---

### 3. Random Forest

El Random Forest se utiliza para modelar relaciones no lineales e interacciones entre los factores experimentales.

El flujo de preprocesamiento incluye one-hot encoding para las variables categóricas y paso directo de las variables numéricas.

La selección de hiperparámetros se realiza con validación cruzada interna sobre los grupos de entrenamiento.

Esta estructura permite comparar la capacidad de generalización de un modelo no lineal frente a la regresión lineal y el baseline.

---

## Métricas de evaluación

Los modelos se comparan mediante tres métricas de regresión:

$$
MAE,
\qquad
RMSE,
\quad
R^2.
$$

El MAE mide el error absoluto promedio.

El RMSE penaliza con mayor intensidad los errores grandes.

Por último, $R^2$ mide qué proporción de la variabilidad de la variable objetivo puede explicar el modelo respecto a una predicción basada únicamente en el promedio.

---

# Resultados del modelado

La comparación entre modelos se realiza sobre particiones agrupadas que mantienen separados los grupos experimentales entre entrenamiento y prueba.

Para reducir la dependencia de una única partición, se utilizan diez holdouts agrupados con semillas distintas.

Los resultados promedio son:

| Modelo | MAE | RMSE | $R^2$ |
|---|---:|---:|---:|
| Baseline de la media | 0.03301 | 0.04365 | -0.0107 |
| Regresión lineal | 0.02745 | 0.03802 | 0.2287 |
| Random Forest | 0.00695 | 0.01009 | 0.9450 |

En este protocolo de evaluación, Random Forest presenta los menores errores y el mayor $R^2$ de los tres modelos comparados.

La diferencia con la regresión lineal sugiere que la relación entre los factores experimentales y $\Delta E_{\text{sync}}$ contiene componentes no lineales e interacciones que un modelo puramente lineal no captura completamente.

---

## 1. Rendimiento comparado

<p align="center">
  <img src="figures/modeling/01_model_performance.png" alt="Comparación de rendimiento entre modelos" width="850">
</p>

*RMSE promedio bajo holdout agrupado repetido y libre de fuga entre grupos experimentales.*

La figura resume el error de predicción de los tres modelos.

El baseline de la media proporciona una referencia mínima: si un modelo supervisado no mejora esta predicción, su utilidad sería limitada.

La regresión lineal reduce el error respecto al baseline, lo que indica que existe una señal predictiva asociada a los factores experimentales.

Random Forest reduce el RMSE de forma mucho más marcada. En las diez particiones agrupadas evaluadas obtuvo un RMSE menor que la regresión lineal y que el baseline.

Para Random Forest, los resultados repetidos presentan aproximadamente

$$
R^2 = 0.9450 \pm 0.0106
$$

y

$$
\mathrm{RMSE}=0.01009 \pm 0.00082.
$$

Estas cifras describen el rendimiento dentro del régimen de interpolación evaluado: los grupos de prueba son nuevos, pero los niveles de los factores experimentales siguen perteneciendo al diseño observado.

---

## 2. Valores observados frente a predichos

<p align="center">
  <img src="figures/modeling/02_rf_predicted_vs_actual.png" alt="Valores observados y predichos por Random Forest" width="780">
</p>

*Predicciones de Random Forest frente a los valores observados de $\Delta E_{\text{sync}}$ en grupos experimentales reservados.*

Cada punto representa una observación del conjunto de prueba.

El eje horizontal contiene el valor observado y el eje vertical la predicción del modelo.

La recta diagonal

$$
\widehat{\Delta E_{\text{sync}}}
=
\Delta E_{\text{sync}}
$$

representa predicción perfecta.

Cuanto más cerca se encuentra un punto de esta diagonal, menor es el error de predicción correspondiente.

La concentración de puntos alrededor de la diagonal es consistente con el alto $R^2$ observado en la evaluación agrupada.

Sin embargo, esta figura debe interpretarse junto con las pruebas de generalización que aparecen más adelante: un buen ajuste sobre combinaciones conocidas del diseño experimental no garantiza extrapolación a estados iniciales completamente nuevos.

---

## 3. Importancia por permutación

<p align="center">
  <img src="figures/modeling/03_rf_permutation_importance.png" alt="Importancia por permutación de las variables del Random Forest" width="850">
</p>

*Incremento del MAE al permutar cada variable en los grupos de prueba.*

La importancia por permutación mide cuánto empeora la predicción cuando se destruye la información de una variable manteniendo las demás sin modificar.

En los experimentos, la importancia predictiva observada sigue aproximadamente el orden:

1. estrategia del actuador;
2. condición inicial;
3. tamaño de la red;
4. acoplamiento $\gamma$;
5. topología.

Esta jerarquía es predictiva y no causal.

Una variable con alta importancia por permutación ayuda al modelo a predecir $\Delta E_{\text{sync}}$, pero esto no demuestra por sí mismo una relación causal ni una prioridad física universal.

---

# Generalización del modelo

Un modelo puede obtener buen rendimiento cuando entrenamiento y prueba contienen niveles conocidos de las variables y, aun así, fallar cuando debe extrapolar hacia una configuración cualitativamente nueva.

Para estudiar esta diferencia se comparan dos regímenes:

- **interpolación:** los grupos de prueba son nuevos, pero pertenecen al mismo espacio factorial observado;
- **extrapolación:** se deja fuera completamente una condición inicial durante el entrenamiento y se evalúa el modelo sobre ella.

---

## 4. Interpolación frente a extrapolación

<p align="center">
  <img src="figures/modeling/04_validation_regimes_v2.png" alt="Comparación entre interpolación y extrapolación del modelo" width="850">
</p>

*Comparación del rendimiento de Random Forest bajo distintos regímenes de validación.*

La validación dejando una condición inicial fuera produce valores de $R^2$ aproximadamente iguales a

| Condición inicial excluida | $R^2$ |
|---:|---:|
| 0 | 0.294 |
| 1 | -0.060 |
| 2 | -2.963 |
| 3 | 0.109 |
| 4 | 0.409 |

El contraste con el $R^2$ cercano a $0.945$ del holdout agrupado muestra que interpolar dentro del diseño observado es considerablemente más sencillo que extrapolar a una condición inicial no vista.

Este resultado también revela una limitación de la representación actual.

La variable

```text
initial_condition_id
```

es una etiqueta categórica. Identifica cada condición inicial, pero no describe directamente sus propiedades físicas o espectrales.

Cuando aparece una condición inicial completamente nueva, el modelo no dispone de una representación continua que le permita relacionarla de manera natural con las condiciones conocidas.

---

## Interpretación metodológica

La caída de rendimiento fuera de las condiciones iniciales observadas no invalida el modelo.

Indica con precisión cuál es su dominio actual de generalización.

Dentro del diseño factorial conocido, Random Forest modela con alta precisión la respuesta de $\Delta E_{\text{sync}}$.

Fuera de ese dominio, especialmente ante condiciones iniciales nuevas, sería necesario construir características más informativas, por ejemplo descriptores derivados del contenido espectral, energía inicial o estructura de los coeficientes de Fourier.

Por tanto, una conclusión central del proyecto es distinguir entre

$$
\boxed{
\text{buen rendimiento de interpolación}
\neq
\text{capacidad automática de extrapolación}
}
$$

Esta distinción evita presentar las métricas del modelo fuera del contexto experimental en el que fueron obtenidas.

---
