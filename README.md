# 🏦 Bank Marketing — Análisis Exploratorio de Datos (EDA)

**Proyecto aplicado — Caso de Estudio N°1**
Especialización en Python for Analytics

**Autor:** Cesar Elías Diaz Marina
**Curso:** Especialización en Python for Analytics
**Año:** 2026

---

## 📌 Descripción del proyecto

Aplicación interactiva construida en **Python + Streamlit** que realiza un
Análisis Exploratorio de Datos (EDA) sobre el dataset `BankMarketing.csv`,
perteneciente a una institución financiera que busca entender qué factores
influyen en la aceptación de sus campañas de marketing telefónico.

El objetivo del proyecto **no** es construir un modelo predictivo, sino aplicar
de forma integrada los conceptos del curso (variables, funciones, f-strings,
POO, NumPy, Pandas, visualización con Matplotlib/Seaborn y estadística
descriptiva) en una herramienta funcional similar a un producto analítico real.

La app está organizada en 4 módulos navegables desde el sidebar:

1. **Home** — presentación del proyecto y datos del autor.
2. **Carga del Dataset** — subida y validación del archivo CSV.
3. **Análisis Exploratorio (EDA)** — 10 ítems de análisis distribuidos en tabs
   (información general, clasificación de variables, estadísticas
   descriptivas, valores faltantes, distribuciones, análisis categórico,
   bivariado numérico-categórico, bivariado categórico-categórico, análisis
   dinámico con parámetros y hallazgos clave).
4. **Conclusiones** — 5 conclusiones orientadas a la toma de decisiones.

## 🧰 Tecnologías utilizadas

- Python 3.10+
- Streamlit
- Pandas / NumPy
- Matplotlib / Seaborn

## 🗂️ Estructura del repositorio

```
├── app.py              # Aplicación principal de Streamlit
├── requirements.txt    # Dependencias del proyecto
├── README.md            # Este archivo
└── BankMarketing.csv   # Dataset utilizado
```

## ▶️ Instrucciones de ejecución

### 1. Clonar el repositorio

```bash
git clone <URL_DEL_REPOSITORIO>
cd <NOMBRE_DEL_REPOSITORIO>
```

### 2. Crear un entorno virtual (opcional pero recomendado)

```bash
python -m venv venv
source venv/bin/activate      # En Windows: venv\Scripts\activate
```

### 3. Instalar las dependencias

```bash
pip install -r requirements.txt
```

### 4. Ejecutar la aplicación

```bash
streamlit run app.py
```

### 5. Cargar el dataset

Dentro de la app, ir al módulo **"Carga del Dataset"** y subir el archivo
`BankMarketing.csv` (el separador utilizado es `;`).

## 🌐 Aplicación desplegada

> 🔗 Reemplazar con el link de Streamlit Cloud una vez publicada:
> `https://<usuario>-<repositorio>.streamlit.app`

## 📸 Capturas de la app

> 🔗 Agregar aquí capturas de pantalla de los módulos Home, Carga del Dataset
> y EDA una vez la app esté desplegada.

## 🔗 Links relevantes

- Repositorio GitHub: `<agregar link>`
- Aplicación desplegada: `<agregar link>`

## 📊 Sobre el dataset

| Variable | Descripción |
|---|---|
| age | Edad del cliente |
| job | Tipo de trabajo del cliente |
| marital | Estado civil |
| education | Nivel educativo |
| default | ¿Tiene crédito en mora? |
| housing | ¿Tiene crédito hipotecario? |
| loan | ¿Tiene crédito personal? |
| contact | Canal de comunicación usado |
| month | Último mes de contacto |
| day_of_week | Día del último contacto |
| duration | Duración del contacto (segundos) |
| campaign | Número de contactos en la campaña actual |
| pdays | Días desde la última gestión |
| previous | Contactos previos antes de la actual campaña |
| poutcome | Resultado de la campaña anterior |
| emp.var.rate | Tasa de variación del empleo |
| cons.price.idx | Índice de precios al consumidor |
| cons.conf.idx | Índice de confianza del consumidor |
| euribor3m | Ratio de tipo de cambio medio (3 meses) |
| nr.employed | Número de empleados |
| y | Resultado final: "yes" si aceptó la campaña, "no" si no la aceptó |
