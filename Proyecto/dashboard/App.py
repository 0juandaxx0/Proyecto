import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

# --- 1. CONFIGURACION DE LA PAGINA ---
st.set_page_config(page_title="Observatorio de Movilidad y Educación", page_icon="🚦", layout="wide")

# --- 2. CARGA DE DATOS Y ENTRENAMIENTO DE MODELO (CACHE) ---
@st.cache_data
def cargar_datos():
    # 1. Datos Universidad
    df_univ = pd.read_csv('../datos_limpios/Datos_universidad.csv')
    map_tiempo = {'0-30 min': 15, '30-60 min': 45, '61-90 min': 75, '91-120 min': 105, 'Mas de 120 min': 135}
    df_univ['minutos_llegada'] = df_univ['tiempo_llegada'].map(map_tiempo)
    df_univ['minutos_regreso'] = df_univ['tiempo_regreso'].map(map_tiempo)
    df_univ['horas_viaje_semanal'] = ((df_univ['minutos_llegada'] + df_univ['minutos_regreso']) * df_univ['dias_asistencia']) / 60
    map_prom = {'3.0 a 3.5': 3.25, '3.6 a 4.0': 3.8, '4.1 a 4.5': 4.3, '4.6 a 5': 4.8}
    df_univ['promedio_num'] = df_univ['promedio_academico'].map(map_prom)
    df_univ['riesgo_academico'] = np.where((df_univ['promedio_num'] <= 3.8) | (df_univ['pierde_clase'] > 0), 1, 0)

    # 2. Datos Abiertos
    df_vel = pd.read_csv('../datos_limpios/Velocidad_promedio.csv')
    df_tiempo = pd.read_csv('../datos_limpios/Tiempo_promedio.csv')
    df_pasajeros = pd.read_csv('../datos_limpios/pasajeros_dia.csv')
    
    return df_univ, df_vel, df_tiempo, df_pasajeros

@st.cache_resource
def entrenar_modelo(df_univ):
    features = ['minutos_llegada', 'trasbordos', 'horas_viaje_semanal', 'dias_asistencia']
    X = df_univ[features].fillna(0)
    y = df_univ['riesgo_academico']
    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    clf.fit(X, y)
    return clf, features

# Cargar todo a memoria
df_univ, df_vel, df_tiempo, df_pasajeros = cargar_datos()
modelo_rf, features_modelo = entrenar_modelo(df_univ)

# --- 3. ENCABEZADO ---
st.title("Observatorio de movilidad y rendimiento academico")
st.markdown("Plataforma interactiva para la toma de decisiones basada en datos abiertos y encuestas universitarias en el corredor Bogota-Soacha.")

# --- 4. CREACION DE PESTAÑAS ---
tab1, tab2, tab3 = st.tabs(["Contexto Ciudad (Movilidad)", "Impacto estudiantil", "Simulador predictivo AI"])

# ==========================================
# PESTAÑA 1: CONTEXTO CIUDAD
# ==========================================
with tab1:
    st.header("Analisis de la Infraestructura Urbana")
    
    # --- KPIs PRINCIPALES ---
    kpi_c1, kpi_c2, kpi_c3 = st.columns(3)
    total_pasajeros_dia = df_pasajeros['pasajeros_dia_tipico_laboral'].max()
    kpi_c1.metric("Volumen Diario (Bogota/Soacha)", f"{total_pasajeros_dia:,.0f}".replace(',', '.'))
    
    vel_actual = df_tiempo[df_tiempo['ano'] == 2025]['velocidad_km/h'].values[0]
    kpi_c2.metric("Velocidad vial actual", f"{vel_actual:.1f} km/h")
    
    df_sitp = pd.read_csv('../datos_limpios/validacion_SITP.csv')
    kpi_c3.metric("Validaciones registradas (Soacha)", f"{len(df_sitp):,.0f}".replace(',', '.'))
    
    st.divider()

    # --- FILA 1: CONTEXTO MACRO (Velocidad y Composición) ---
    col_c1, col_c2 = st.columns(2)
    
    with col_c1:
        # ==========================================
        st.subheader("Caída historica de la velocidad")
        st.markdown("Evolución del trafico vehicular en las ultimas dos decadas sin duplicados.")

        # 1. Extraer y renombrar columnas
        hist_vel = df_vel[['ano', 'velocidad_promedio_km/h']].rename(columns={'velocidad_promedio_km/h': 'Velocidad'})
        rec_vel = df_tiempo[['ano', 'velocidad_km/h']].rename(columns={'velocidad_km/h': 'Velocidad'})

        # 2. Combinar los conjuntos de datos
        df_vel_total = pd.concat([hist_vel, rec_vel])

        # 3. SOLUCION CRITICA: Eliminar duplicados de años, asegurar tipo entero y ordenar
        df_vel_total = df_vel_total.drop_duplicates(subset=['ano'], keep='last')
        df_vel_total['ano'] = df_vel_total['ano'].astype(int)
        df_vel_total = df_vel_total.sort_values('ano')

        # 4. Crear el grafico configurando de forma explicita el eje X como lineal continuo
        fig_vel = px.line(df_vel_total, x='ano', y='Velocidad', markers=True)

        # Ajustes de diseño para que se vea limpio, espaciado y profesional
        fig_vel.update_layout(
            xaxis=dict(
                type='linear',
                tickmode='linear',
                dtick=2,  # Muestra etiquetas cada 2 años de forma uniforme
                title="Año"
            ),
            yaxis=dict(title="Velocidad (km/h)"),
            margin=dict(t=20, b=20, l=20, r=20)
        )
        fig_vel.update_traces(line_width=3, marker=dict(size=8))
        st.plotly_chart(fig_vel, use_container_width=True)
        
    with col_c2:
        st.subheader("Carga del sistema de transporte")
        st.markdown("Distribución diaria de los 3.9 millones de pasajeros.")
        
        # Procesamos pasajeros_dia para sacar troncal vs zonal
        valores_unicos = df_pasajeros['pasajeros_dia_tipico_laboral'].unique()
        valores_unicos.sort() # Ordenar de menor a mayor: [1426622, 2517676, 3944298]
        
        if len(valores_unicos) >= 3:
            datos_comp = pd.DataFrame({
                'Componente': ['SITP / Zonal (Buses)', 'TransMilenio / Troncal'],
                'Pasajeros': [valores_unicos[0], valores_unicos[1]]
            })
            
            # Grafico de Donas (Donut chart)
            fig_comp = px.pie(datos_comp, names='Componente', values='Pasajeros', hole=0.5,
                              color_discrete_sequence=['#1f77b4', '#d62728']) # Azul SITP y Rojo TM
            
            # Mejoramos el diseño para que se lea claro y profesional
            fig_comp.update_traces(textposition='inside', textinfo='percent+label', 
                                   hovertemplate="%{label}: <br>%{value:,.0f} pasajeros")
            fig_comp.update_layout(showlegend=False, margin=dict(t=30, b=0, l=0, r=0))
            st.plotly_chart(fig_comp, use_container_width=True)

    st.divider()

    # --- FILA 2: MICRO CONTEXTO SOACHA (Cuellos de Botella) ---
    st.subheader("Presion del sistema (Cuellos de botella en Soacha)")
    st.markdown("Volumen de validaciones reales por estacion y punto de acceso.")
    
    congest_estacion = df_sitp.groupby(['estacion_parada', 'acceso_estacion']).size().reset_index(name='cantidad')
    congest_estacion = congest_estacion.sort_values(by='cantidad', ascending=False)

    fig_presion = px.bar(
        congest_estacion, 
        x='estacion_parada', 
        y='cantidad', 
        color='acceso_estacion',
        color_discrete_sequence=px.colors.qualitative.Bold,
        labels={'estacion_parada': 'Estación', 'cantidad': 'Volumen de Ingresos', 'acceso_estacion': 'Punto de Acceso'}
    )
    fig_presion.update_layout(barmode='stack', legend=dict(orientation="h", yanchor="bottom", y=-0.5, xanchor="center", x=0.5))
    
    # Usamos container width para que abarque toda la pantalla horizontalmente
    st.plotly_chart(fig_presion, use_container_width=True)

# ==========================================
# PESTAÑA 2: IMPACTO ESTUDIANTIL
# ==========================================
with tab2:
    st.header("Realidad universitaria")
    
    # KPIs rápidos
    st.markdown("#### 📊 Indicadores clave")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Promedio horas viaje/semana", f"{df_univ['horas_viaje_semanal'].mean():.1f} h")
    kpi2.metric("Llegadas tarde promedio", f"{df_univ['llegadas_tarde'].mean():.1f}")
    kpi3.metric("Clases perdidas promedio", f"{df_univ['pierde_clase'].mean():.1f}")
    porc_riesgo = (df_univ['riesgo_academico'].sum() / len(df_univ)) * 100
    kpi4.metric("Estudiantes en riesgo alto", f"{porc_riesgo:.1f}%", delta="Crítico", delta_color="inverse")
    
    st.divider()
    
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.markdown("**Distribucion por medio de transporte**")
        fig_trans = px.pie(df_univ, names='medio_transporte', hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_trans, use_container_width=True)
        
    with col_e2:
        st.markdown("**Tiempo de viaje vs clases perdidas**")
        fig_box = px.box(df_univ, x='tiempo_llegada', y='pierde_clase', color='tiempo_llegada',
                         category_orders={"tiempo_llegada": ['0-30 min', '30-60 min', '61-90 min', '91-120 min', 'Mas de 120 min']})
        st.plotly_chart(fig_box, use_container_width=True)

# ==========================================
# PESTAÑA 3: SIMULADOR DE RIESGO (MACHINE LEARNING)
# ==========================================
with tab3:
    st.header("Simulador predictivo de riesgo academico")
    st.markdown("""
    Esta herramienta utiliza un modelo de **Random Forest Classifier** entrenado con los datos de la encuesta. 
    Ajusta los parámetros de movilidad de un estudiante hipotético para estimar la probabilidad de que empiece a faltar a clases o baje su promedio.
    """)
    
    st.divider()
    
    col_sim1, col_sim2 = st.columns([1, 1])
    
    with col_sim1:
        st.subheader("Parametros del estudiante")
        sim_minutos = st.slider("Minutos de viaje (Solo ida)", min_value=10, max_value=180, value=75, step=5)
        sim_trasbordos = st.slider("Cantidad de trasbordos (Ida)", min_value=0, max_value=4, value=1)
        sim_dias = st.slider("Días de asistencia a la semana", min_value=1, max_value=6, value=5)
        
        # Calculo automatico de horas semanales basado en los inputs
        sim_horas_semana = ((sim_minutos * 2) * sim_dias) / 60
        st.info(f"Tiempo total invertido a la semana: **{sim_horas_semana:.1f} horas**")
        
    with col_sim2:
        st.subheader("Prediccion del modelo")
        
        # Crear array con los datos del usuario
        input_data = pd.DataFrame([[sim_minutos, sim_trasbordos, sim_horas_semana, sim_dias]], columns=features_modelo)
        
        # Predecir probabilidad
        probabilidad = modelo_rf.predict_proba(input_data)[0][1] * 100
        
        # Grafico tipo velocimetro
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = probabilidad,
            number = {'suffix': "%"},
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Probabilidad de riesgo academico", 'font': {'size': 20}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "rgba(0,0,0,0)"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 33], 'color': "lightgreen"},
                    {'range': [33, 66], 'color': "gold"},
                    {'range': [66, 100], 'color': "salmon"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': probabilidad}
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        if probabilidad < 33:
            st.success("Estudiante en zona segura. El impacto logistico es manejable.")
        elif probabilidad < 66:
            st.warning("Riesgo moderado. El estudiante podria empezar a experimentar fatiga y fallas esporadicas.")
        else:
            st.error("Riesgo critico. Alta probabilidad de inasistencia recurrente o caida en el promedio. Se sugieren medidas de apoyo.")

# --- FOOTER ---
st.divider()
st.caption("Desarrollado para la toma de decisiones institucionales y análisis de movilidad urbana.")