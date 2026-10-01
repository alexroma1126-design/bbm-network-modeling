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
