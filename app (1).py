"""
================================================================================
 APLICACIÓN: Análisis Exploratorio de Datos - BankMarketing
 CURSO: Especialización en Python for Analytics
 AUTOR: Cesar Elías Diaz Marina
 AÑO: 2026
================================================================================
Este proyecto integra: variables y tipos de datos, funciones, f-strings,
Programación Orientada a Objetos (POO), NumPy, Pandas, visualización con
Matplotlib/Seaborn y estadística descriptiva, dentro de una app Streamlit.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------------------------
# CONFIGURACIÓN GENERAL DE LA PÁGINA
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Bank Marketing - EDA",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

sns.set_theme(style="whitegrid")

# ------------------------------------------------------------------------------
# DATOS DEL AUTOR (variables y f-strings)
# ------------------------------------------------------------------------------
AUTOR_NOMBRE = "Cesar Elías Diaz Marina"
AUTOR_CURSO = "Especialización en Python for Analytics"
AUTOR_ANIO = 2026


# ==============================================================================
# FUNCIÓN PERSONALIZADA: clasificación de una variable individual
# (Requisito del Ítem 2 -> "uso de una función personalizada")
# ==============================================================================
def clasificar_tipo_variable(serie: pd.Series) -> str:
    """
    Clasifica una columna de un DataFrame como 'Numérica' o 'Categórica'
    en función de su tipo de dato de pandas.

    Parámetros
    ----------
    serie : pd.Series
        Columna a clasificar.

    Retorna
    -------
    str
        'Numérica' o 'Categórica'
    """
    if pd.api.types.is_numeric_dtype(serie):
        return "Numérica"
    return "Categórica"


# ==============================================================================
# CLASE: DataAnalyzer (Programación Orientada a Objetos)
# Encapsula estadísticas descriptivas, clasificación de variables y
# funciones de visualización, tal como exige el caso de estudio.
# ==============================================================================
class DataAnalyzer:
    """Clase que encapsula la lógica de análisis exploratorio del dataset."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.n_filas, self.n_columnas = df.shape

    # ---------------------------------------------------------------- INFO ---
    def resumen_general(self) -> pd.DataFrame:
        """Devuelve un resumen con tipo de dato y conteo de nulos por columna."""
        resumen = pd.DataFrame({
            "Columna": self.df.columns,
            "Tipo de dato": self.df.dtypes.values.astype(str),
            "Valores nulos": self.df.isnull().sum().values,
            "% Nulos": (self.df.isnull().sum().values / self.n_filas * 100).round(2),
        })
        return resumen

    # ---------------------------------------------------- CLASIFICACIÓN ---
    def clasificar_variables(self) -> dict:
        """
        Usa la función personalizada clasificar_tipo_variable() para separar
        las columnas del DataFrame en numéricas y categóricas.
        """
        numericas, categoricas = [], []
        for col in self.df.columns:
            tipo = clasificar_tipo_variable(self.df[col])
            if tipo == "Numérica":
                numericas.append(col)
            else:
                categoricas.append(col)
        return {"numericas": numericas, "categoricas": categoricas}

    # ------------------------------------------------- ESTADÍSTICA DESC. ---
    def estadisticas_descriptivas(self, columnas=None) -> pd.DataFrame:
        """Retorna .describe() para las columnas numéricas indicadas."""
        if columnas is None:
            columnas = self.df.select_dtypes(include=np.number).columns.tolist()
        return self.df[columnas].describe().T

    def medidas_resumen(self, columna: str) -> dict:
        """Calcula media, mediana, moda y desviación estándar de una columna numérica."""
        serie = self.df[columna].dropna()
        moda = serie.mode()
        return {
            "media": serie.mean(),
            "mediana": serie.median(),
            "moda": moda.iloc[0] if not moda.empty else np.nan,
            "desviacion_std": serie.std(),
            "minimo": serie.min(),
            "maximo": serie.max(),
        }

    # ---------------------------------------------------- VALORES NULOS ---
    def conteo_nulos(self) -> pd.Series:
        return self.df.isnull().sum().sort_values(ascending=False)

    # ------------------------------------------------------ VISUALIZACIÓN ---
    def graficar_histograma(self, columna: str, bins: int = 30):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.histplot(self.df[columna].dropna(), bins=bins, kde=True, color="#2563eb", ax=ax)
        ax.set_title(f"Distribución de '{columna}'", fontsize=13, fontweight="bold")
        ax.set_xlabel(columna)
        ax.set_ylabel("Frecuencia")
        fig.tight_layout()
        return fig

    def graficar_barras_categorica(self, columna: str, top_n: int = 15):
        conteo = self.df[columna].value_counts().head(top_n)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.barplot(x=conteo.values, y=conteo.index, hue=conteo.index,
                    palette="Blues_r", legend=False, ax=ax)
        ax.set_title(f"Conteo de categorías - '{columna}'", fontsize=13, fontweight="bold")
        ax.set_xlabel("Frecuencia")
        ax.set_ylabel(columna)
        fig.tight_layout()
        return fig

    def graficar_bivariado_num_cat(self, num_col: str, cat_col: str):
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        sns.boxplot(data=self.df, x=cat_col, y=num_col, hue=cat_col,
                    palette="viridis", legend=False, ax=ax)
        ax.set_title(f"'{num_col}' según '{cat_col}'", fontsize=13, fontweight="bold")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        return fig

    def graficar_bivariado_cat_cat(self, cat_col1: str, cat_col2: str):
        tabla = pd.crosstab(self.df[cat_col1], self.df[cat_col2], normalize="index") * 100
        fig, ax = plt.subplots(figsize=(8, 4.8))
        tabla.plot(kind="bar", stacked=True, colormap="coolwarm", ax=ax)
        ax.set_title(f"'{cat_col2}' según '{cat_col1}' (%)", fontsize=13, fontweight="bold")
        ax.set_ylabel("Porcentaje (%)")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(title=cat_col2, bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        return fig

    def graficar_correlacion(self, columnas):
        fig, ax = plt.subplots(figsize=(7, 5.5))
        corr = self.df[columnas].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Matriz de correlación", fontsize=13, fontweight="bold")
        fig.tight_layout()
        return fig


# ==============================================================================
# BARRA LATERAL - MENÚ DE NAVEGACIÓN
# ==============================================================================
st.sidebar.title("🏦 Bank Marketing App")
st.sidebar.markdown("---")
modulo = st.sidebar.radio(
    "Navegación",
    ["🏠 Home", "📁 Carga del Dataset", "🔎 Análisis Exploratorio (EDA)", "📌 Conclusiones"],
)
st.sidebar.markdown("---")
st.sidebar.caption(f"👤 {AUTOR_NOMBRE}")
st.sidebar.caption(f"🎓 {AUTOR_CURSO}")
st.sidebar.caption(f"📅 {AUTOR_ANIO}")

# Estado de sesión para conservar el dataset cargado entre módulos
if "df" not in st.session_state:
    st.session_state.df = None


# ==============================================================================
# MÓDULO 1: HOME
# ==============================================================================
if modulo == "🏠 Home":
    st.title("🏦 Análisis Exploratorio de Datos: Bank Marketing")
    st.markdown("#### Proyecto aplicado - Especialización en Python for Analytics")
    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("🎯 Objetivo del análisis")
        st.write(
            f"""
            Esta aplicación interactiva desarrolla un **Análisis Exploratorio de Datos (EDA)**
            sobre el dataset `BankMarketing.csv`, perteneciente a una institución financiera
            que busca entender qué factores influyen en la aceptación de sus campañas de
            marketing telefónico.

            Durante los últimos 6 meses la efectividad de la campaña (Ventas/Base) cayó de
            **12% a 8%**, afectando los bonos de los ejecutivos comerciales. El objetivo de
            este proyecto **no es construir un modelo predictivo**, sino explorar los datos
            para identificar relaciones y comportamientos relevantes que apoyen la
            **toma de decisiones** del equipo comercial.
            """
        )

        st.subheader("🧰 Tecnologías utilizadas")
        st.markdown(
            """
            - **Python** — lenguaje base del proyecto
            - **Pandas / NumPy** — manipulación y análisis de datos
            - **Matplotlib / Seaborn** — visualización de datos
            - **Streamlit** — construcción de la interfaz interactiva
            - **Programación Orientada a Objetos (POO)** — clase `DataAnalyzer`
            """
        )

    with col2:
        st.subheader("👤 Datos del autor")
        st.info(
            f"""
            **Nombre:** {AUTOR_NOMBRE}

            **Curso:** {AUTOR_CURSO}

            **Año:** {AUTOR_ANIO}
            """
        )
        st.subheader("📦 Sobre el dataset")
        st.write(
            """
            El dataset contiene **21 variables** relacionadas con datos demográficos del
            cliente (edad, ocupación, estado civil, educación), su situación financiera
            (crédito en mora, hipoteca, préstamo personal), detalles del contacto de la
            campaña (canal, mes, duración) e indicadores macroeconómicos
            (tasa de empleo, euribor, índice de confianza del consumidor).

            La variable objetivo es **`y`**: indica si el cliente aceptó ("yes") o no
            ("no") la oferta de la campaña.
            """
        )

    st.markdown("---")
    st.caption(
        "👉 Dirígete al módulo **Carga del Dataset** en la barra lateral para comenzar el análisis."
    )


# ==============================================================================
# MÓDULO 2: CARGA DEL DATASET
# ==============================================================================
elif modulo == "📁 Carga del Dataset":
    st.title("📁 Carga del Dataset")
    st.write(
        "Sube el archivo `BankMarketing.csv` para habilitar el módulo de Análisis "
        "Exploratorio de Datos. El archivo utiliza `;` como separador."
    )

    archivo = st.file_uploader("Selecciona el archivo CSV", type=["csv"])

    if archivo is not None:
        try:
            df_cargado = pd.read_csv(archivo, sep=";")
            st.session_state.df = df_cargado
            st.success(f"✅ Archivo **{archivo.name}** cargado correctamente.")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Número de filas", f"{df_cargado.shape[0]:,}")
            with col2:
                st.metric("Número de columnas", f"{df_cargado.shape[1]}")

            st.subheader("Vista previa del dataset")
            st.dataframe(df_cargado.head(10), use_container_width=True)

        except Exception as e:
            st.error(f"❌ Ocurrió un error al leer el archivo: {e}")
            st.session_state.df = None
    else:
        st.warning("⚠️ Aún no se ha cargado ningún archivo. El módulo de EDA permanecerá bloqueado.")


# ==============================================================================
# MÓDULO 3: ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# ==============================================================================
elif modulo == "🔎 Análisis Exploratorio (EDA)":
    st.title("🔎 Análisis Exploratorio de Datos (EDA)")

    if st.session_state.df is None:
        st.error("🚫 Debes cargar el dataset primero en el módulo **Carga del Dataset**.")
        st.stop()

    df = st.session_state.df
    analyzer = DataAnalyzer(df)
    clasificacion = analyzer.clasificar_variables()
    num_cols = clasificacion["numericas"]
    cat_cols = clasificacion["categoricas"]

    tabs = st.tabs([
        "1️⃣ Info general", "2️⃣ Clasificación", "3️⃣ Estadísticas", "4️⃣ Nulos",
        "5️⃣ Distribuciones", "6️⃣ Categóricas", "7️⃣ Bivariado num-cat",
        "8️⃣ Bivariado cat-cat", "9️⃣ Análisis dinámico", "🔟 Hallazgos clave",
    ])

    # ---------------------------- ÍTEM 1: INFORMACIÓN GENERAL ----------------------------
    with tabs[0]:
        st.subheader("Información general del dataset")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Filas", f"{analyzer.n_filas:,}")
        with c2:
            st.metric("Columnas", analyzer.n_columnas)

        st.write("**Resumen de tipos de datos y valores nulos por columna:**")
        st.dataframe(analyzer.resumen_general(), use_container_width=True)

        if st.checkbox("Mostrar salida técnica de .info()"):
            import io
            buffer = io.StringIO()
            df.info(buf=buffer)
            st.text(buffer.getvalue())

    # ---------------------------- ÍTEM 2: CLASIFICACIÓN DE VARIABLES ----------------------------
    with tabs[1]:
        st.subheader("Clasificación de variables (función personalizada)")
        st.write(
            "Se aplicó la función `clasificar_tipo_variable()` a cada columna del dataset "
            "para separar las variables en **numéricas** y **categóricas**."
        )
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Variables numéricas ({len(num_cols)})**")
            st.write(num_cols)
        with c2:
            st.markdown(f"**Variables categóricas ({len(cat_cols)})**")
            st.write(cat_cols)

        conteo_tipo = pd.Series(
            {"Numéricas": len(num_cols), "Categóricas": len(cat_cols)}
        )
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.barplot(x=conteo_tipo.index, y=conteo_tipo.values, hue=conteo_tipo.index,
                    palette="Set2", legend=False, ax=ax)
        ax.set_ylabel("Cantidad de variables")
        fig.tight_layout()
        st.pyplot(fig)

    # ---------------------------- ÍTEM 3: ESTADÍSTICAS DESCRIPTIVAS ----------------------------
    with tabs[2]:
        st.subheader("Estadísticas descriptivas")
        cols_sel = st.multiselect(
            "Selecciona variables numéricas a describir", num_cols, default=num_cols[:5]
        )
        if cols_sel:
            st.dataframe(analyzer.estadisticas_descriptivas(cols_sel), use_container_width=True)

            col_focus = st.selectbox("Interpretar en detalle la variable:", cols_sel)
            medidas = analyzer.medidas_resumen(col_focus)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Media", f"{medidas['media']:.2f}")
            c2.metric("Mediana", f"{medidas['mediana']:.2f}")
            c3.metric("Moda", f"{medidas['moda']:.2f}")
            c4.metric("Desv. estándar", f"{medidas['desviacion_std']:.2f}")

            st.write(
                f"""
                Para **`{col_focus}`**, la media ({medidas['media']:.2f}) y la mediana
                ({medidas['mediana']:.2f}) {"son cercanas, lo que sugiere una distribución "
                "relativamente simétrica" if abs(medidas['media'] - medidas['mediana']) < medidas['desviacion_std'] * 0.3
                else "difieren de forma notoria, lo que sugiere una distribución sesgada"}.
                El rango de valores va de {medidas['minimo']:.2f} a {medidas['maximo']:.2f},
                con una dispersión (desviación estándar) de {medidas['desviacion_std']:.2f}.
                """
            )
        else:
            st.info("Selecciona al menos una variable numérica.")

    # ---------------------------- ÍTEM 4: VALORES FALTANTES ----------------------------
    with tabs[3]:
        st.subheader("Análisis de valores faltantes")
        nulos = analyzer.conteo_nulos()
        nulos_con_valor = nulos[nulos > 0]

        st.dataframe(nulos.to_frame("Valores nulos"), use_container_width=True)

        if not nulos_con_valor.empty:
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.barplot(x=nulos_con_valor.values, y=nulos_con_valor.index,
                        hue=nulos_con_valor.index, palette="Reds_r", legend=False, ax=ax)
            ax.set_xlabel("Cantidad de nulos")
            fig.tight_layout()
            st.pyplot(fig)
            st.write(
                "Se identificaron columnas con valores nulos explícitos. Se recomienda "
                "evaluar si deben imputarse o excluirse antes de un análisis más profundo."
            )
        else:
            st.success(
                "✅ No se detectaron valores nulos explícitos (NaN). Sin embargo, algunas "
                "columnas categóricas usan la etiqueta `'unknown'` como valor faltante "
                "implícito, lo cual se recomienda revisar por separado."
            )
            if "job" in df.columns:
                unknowns = (df.select_dtypes(include=["object", "string"]) == "unknown").sum()
                unknowns = unknowns[unknowns > 0].sort_values(ascending=False)
                if not unknowns.empty:
                    st.write("**Conteo de valores `'unknown'` por columna categórica:**")
                    st.dataframe(unknowns.to_frame("Conteo 'unknown'"), use_container_width=True)

    # ---------------------------- ÍTEM 5: DISTRIBUCIÓN DE VARIABLES NUMÉRICAS ----------------------------
    with tabs[4]:
        st.subheader("Distribución de variables numéricas")
        col1, col2 = st.columns([1, 2])
        with col1:
            var_num = st.selectbox("Variable numérica", num_cols, key="hist_var")
            bins = st.slider("Número de bins", min_value=5, max_value=100, value=30, step=5)
        with col2:
            fig = analyzer.graficar_histograma(var_num, bins)
            st.pyplot(fig)
        st.caption(
            f"El histograma de **{var_num}** permite observar la forma de la distribución "
            "(simetría, sesgo, presencia de valores atípicos) y su concentración de datos."
        )

    # ---------------------------- ÍTEM 6: ANÁLISIS DE VARIABLES CATEGÓRICAS ----------------------------
    with tabs[5]:
        st.subheader("Análisis de variables categóricas")
        col1, col2 = st.columns([1, 2])
        with col1:
            var_cat = st.selectbox("Variable categórica", cat_cols, key="cat_var")
            mostrar_pct = st.checkbox("Mostrar proporciones (%)", value=True)
        with col2:
            fig = analyzer.graficar_barras_categorica(var_cat)
            st.pyplot(fig)

        if mostrar_pct:
            proporciones = (df[var_cat].value_counts(normalize=True) * 100).round(2)
            st.write("**Proporciones (%):**")
            st.dataframe(proporciones.to_frame("Porcentaje"), use_container_width=True)

    # ---------------------------- ÍTEM 7: BIVARIADO NUMÉRICO VS CATEGÓRICO ----------------------------
    with tabs[6]:
        st.subheader("Análisis bivariado: numérico vs categórico")
        c1, c2 = st.columns(2)
        with c1:
            num_biv = st.selectbox("Variable numérica", num_cols,
                                    index=num_cols.index("age") if "age" in num_cols else 0,
                                    key="biv_num")
        with c2:
            cat_biv = st.selectbox("Variable categórica", cat_cols,
                                    index=cat_cols.index("y") if "y" in cat_cols else 0,
                                    key="biv_cat")
        fig = analyzer.graficar_bivariado_num_cat(num_biv, cat_biv)
        st.pyplot(fig)
        st.caption(
            f"El boxplot compara la distribución de **{num_biv}** entre las distintas "
            f"categorías de **{cat_biv}**, permitiendo identificar diferencias de mediana, "
            "dispersión y posibles valores atípicos entre grupos."
        )

    # ---------------------------- ÍTEM 8: BIVARIADO CATEGÓRICO VS CATEGÓRICO ----------------------------
    with tabs[7]:
        st.subheader("Análisis bivariado: categórico vs categórico")
        c1, c2 = st.columns(2)
        with c1:
            cat1 = st.selectbox("Variable categórica 1 (eje X)", cat_cols,
                                 index=cat_cols.index("education") if "education" in cat_cols else 0,
                                 key="biv_cat1")
        with c2:
            opciones_cat2 = [c for c in cat_cols if c != cat1]
            cat2 = st.selectbox("Variable categórica 2 (segmentación)", opciones_cat2,
                                 index=opciones_cat2.index("y") if "y" in opciones_cat2 else 0,
                                 key="biv_cat2")
        fig = analyzer.graficar_bivariado_cat_cat(cat1, cat2)
        st.pyplot(fig)
        st.caption(
            f"El gráfico de barras apiladas muestra cómo se distribuye **{cat2}** "
            f"(en porcentaje) dentro de cada categoría de **{cat1}**."
        )

    # ---------------------------- ÍTEM 9: ANÁLISIS BASADO EN PARÁMETROS SELECCIONADOS ----------------------------
    with tabs[8]:
        st.subheader("Análisis dinámico según parámetros seleccionados")
        st.write("Filtra y explora el dataset ajustando los parámetros a continuación.")

        c1, c2, c3 = st.columns(3)
        with c1:
            edad_min, edad_max = int(df["age"].min()), int(df["age"].max())
            rango_edad = st.slider("Rango de edad", edad_min, edad_max, (edad_min, edad_max))
        with c2:
            trabajos_sel = st.multiselect(
                "Filtrar por ocupación (job)", sorted(df["job"].unique()),
                default=sorted(df["job"].unique())[:3] if "job" in df.columns else []
            )
        with c3:
            mostrar_datos = st.checkbox("Mostrar tabla de datos filtrados")

        df_filtrado = df[(df["age"] >= rango_edad[0]) & (df["age"] <= rango_edad[1])]
        if trabajos_sel:
            df_filtrado = df_filtrado[df_filtrado["job"].isin(trabajos_sel)]

        st.metric("Registros tras el filtro", f"{len(df_filtrado):,}")

        if len(df_filtrado) > 0 and "y" in df_filtrado.columns:
            tasa_aceptacion = (df_filtrado["y"] == "yes").mean() * 100
            st.metric("Tasa de aceptación de la campaña (%)", f"{tasa_aceptacion:.2f}%")

            analyzer_filtrado = DataAnalyzer(df_filtrado)
            fig = analyzer_filtrado.graficar_histograma("age", bins=25)
            st.pyplot(fig)

        if mostrar_datos:
            st.dataframe(df_filtrado, use_container_width=True)

    # ---------------------------- ÍTEM 10: HALLAZGOS CLAVE ----------------------------
    with tabs[9]:
        st.subheader("Hallazgos clave del análisis")

        if "y" in df.columns:
            tasa_global = (df["y"] == "yes").mean() * 100
            st.metric("Tasa de aceptación global de la campaña", f"{tasa_global:.2f}%")

        cols_corr = [c for c in ["age", "duration", "campaign", "pdays", "previous",
                                  "emp.var.rate", "euribor3m", "nr.employed"] if c in num_cols]
        if len(cols_corr) >= 2:
            st.write("**Matriz de correlación entre variables numéricas relevantes:**")
            fig = analyzer.graficar_correlacion(cols_corr)
            st.pyplot(fig)

        st.markdown("#### 💡 Insights principales")
        st.markdown(
            """
            - La **duración de la llamada** (`duration`) muestra una relación notoria con la
              aceptación de la campaña: contactos más largos tienden a asociarse con mayor
              probabilidad de éxito.
            - Existen diferencias claras en la tasa de aceptación según el **canal de
              contacto** (`contact`) y el **nivel educativo** (`education`).
            - Los indicadores macroeconómicos (`euribor3m`, `emp.var.rate`, `nr.employed`)
              están fuertemente correlacionados entre sí, lo que sugiere que reflejan un
              mismo ciclo económico.
            - Un número elevado de contactos previos sin éxito (`poutcome = failure`) se
              relaciona con menor probabilidad de aceptación en la campaña actual.
            """
        )


# ==============================================================================
# MÓDULO 4: CONCLUSIONES
# ==============================================================================
elif modulo == "📌 Conclusiones":
    st.title("📌 Conclusiones Finales")

    if st.session_state.df is None:
        st.warning(
            "⚠️ Aún no has cargado el dataset. Puedes revisar las conclusiones generales, "
            "pero se recomienda explorar el módulo de EDA primero para contextualizarlas."
        )

    st.markdown(
        """
        1. **La duración de la llamada es el indicador más determinante del éxito de la
           campaña.** Los clientes con contactos más largos muestran una probabilidad
           notablemente mayor de aceptar la oferta, lo que sugiere priorizar la calidad de
           la conversación sobre la cantidad de contactos.

        2. **El contexto macroeconómico condiciona fuertemente el resultado comercial.**
           Variables como el euribor a 3 meses, la tasa de variación del empleo y el número
           de empleados están altamente correlacionadas y explican parte de la caída de
           efectividad observada en los últimos meses.

        3. **El histórico de contacto con el cliente importa.** Los clientes con resultado
           positivo en campañas anteriores (`poutcome = success`) presentan una tasa de
           aceptación considerablemente más alta que aquellos sin contacto previo o con
           resultado negativo.

        4. **El perfil demográfico influye, pero de forma secundaria.** Variables como edad
           y nivel educativo muestran diferencias moderadas en la tasa de aceptación,
           útiles para segmentar campañas, pero no son, por sí solas, predictoras fuertes.

        5. **La calidad de los datos es adecuada pero requiere atención en variables
           categóricas.** Aunque no existen valores nulos explícitos, la presencia de la
           categoría `'unknown'` en varias columnas debe tratarse como información
           faltante al momento de tomar decisiones basadas en estos datos.
        """
    )

    st.markdown("---")
    st.info(
        "Estas conclusiones están orientadas a apoyar la **toma de decisiones** del equipo "
        "comercial (por ejemplo, priorizar canales, horarios y perfiles de cliente), y no "
        "constituyen un modelo predictivo."
    )
