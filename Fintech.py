import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Configuración inicial
st.set_page_config(page_title="Finanzas Personales", page_icon="💰", layout="wide")

# Lista inicial de formas de pago
formas_pago = ['transferencia', 'depósito', 'efectivo']

# Funciones de Análisis Avanzado

def realizar_pca_finanzas(df):
    """
    Realiza Análisis de Componentes Principales en datos financieros
    """
    try:
        # Seleccionar columnas numéricas para PCA
        columnas_numericas = df.select_dtypes(include=[np.number]).columns
        
        # Preparar los datos
        X = df[columnas_numericas]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Realizar PCA
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)
        
        # Calcular la varianza explicada
        varianza_explicada = pca.explained_variance_ratio_
        varianza_acumulada = np.cumsum(varianza_explicada)
        
        # Visualizar varianza explicada
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, len(varianza_explicada) + 1), varianza_explicada, alpha=0.5, align='center',
                label='Varianza Individual Explicada')
        plt.step(range(1, len(varianza_acumulada) + 1), varianza_acumulada, where='mid',
                 label='Varianza Acumulada')
        plt.ylabel('Ratio de Varianza Explicada')
        plt.xlabel('Componentes Principales')
        plt.title('Análisis de Varianza en Componentes Principales')
        plt.legend(loc='best')
        plt.tight_layout()
        
        # Crear diccionario de resultados
        resultados_pca = {
            'varianza_explicada': varianza_explicada,
            'varianza_acumulada': varianza_acumulada,
            'componentes_principales': pca.components_,
            'grafico_varianza': plt
        }
        
        return resultados_pca
    except Exception as e:
        st.error(f"Error en PCA: {e}")
        return None

def interpretar_pca(resultados_pca):
    """
    Interpreta los resultados del Análisis de Componentes Principales
    """
    insights = []
    
    # Umbral de varianza explicada significativa
    umbral_varianza = 0.80
    
    # Número de componentes necesarios para explicar el umbral de varianza
    componentes_necesarios = np.argmax(resultados_pca['varianza_acumulada'] >= umbral_varianza) + 1
    
    # Generar insights
    insights.append(f"Se necesitan {componentes_necesarios} componentes principales para explicar el {umbral_varianza*100:.2f}% de la variabilidad en tus finanzas.")
    
    # Analizar la varianza explicada por cada componente
    for i, varianza in enumerate(resultados_pca['varianza_explicada'][:3], 1):
        insights.append(f"Componente Principal {i} explica {varianza*100:.2f}% de la varianza en tus datos financieros.")
    
    return insights

def analisis_correlacion_financiera(df):
    """
    Análisis de correlación entre variables financieras
    """
    try:
        # Seleccionar columnas numéricas
        columnas_numericas = df.select_dtypes(include=[np.number]).columns
        
        # Crear matriz de correlación
        matriz_correlacion = df[columnas_numericas].corr()
        
        # Visualizar heatmap de correlaciones
        plt.figure(figsize=(10, 8))
        sns.heatmap(matriz_correlacion, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlaciones entre Variables Financieras')
        st.pyplot(plt)
    except Exception as e:
        st.error(f"Error en análisis de correlación: {e}")

def transformaciones_lineales(df):
    """
    Transformaciones lineales de gastos
    """
    try:
        # Calcular transformaciones lineales de gastos
        df['gasto_normalizado'] = (df['Monto'] - df['Monto'].mean()) / df['Monto'].std()
        
        # Identificar outliers usando transformaciones lineales
        outliers = df[np.abs(df['gasto_normalizado']) > 2]
        
        st.subheader("🔍 Análisis de Transacciones Atípicas")
        if not outliers.empty:
            st.write("Transacciones atípicas detectadas:")
            st.dataframe(outliers)
        else:
            st.write("No se encontraron transacciones atípicas significativas.")
    except Exception as e:
        st.error(f"Error en transformaciones lineales: {e}")

def analisis_vectorial_finanzas(df):
    """
    Análisis vectorial de gastos
    """
    try:
        # Descomponer gastos en componentes
        gastos_vector = df[df['Tipo'] == 'Gasto']['Monto'].values
        
        # Calcular magnitud y dirección de gastos
        magnitud_gastos = np.linalg.norm(gastos_vector)
        
        st.subheader("📊 Análisis Vectorial de Gastos")
        st.write(f"**Magnitud total de gastos:** ${magnitud_gastos:.2f}")
        
        # Distribución de gastos por categoría
        gastos_por_categoria = df[df['Tipo'] == 'Gasto'].groupby('Descripción')['Monto'].sum()
        
        plt.figure(figsize=(10, 6))
        gastos_por_categoria.plot(kind='pie', autopct='%1.1f%%')
        plt.title('Distribución de Gastos por Categoría')
        st.pyplot(plt)
    except Exception as e:
        st.error(f"Error en análisis vectorial: {e}")

def mostrar_analisis(df):
    """
    Función principal de análisis de datos financieros
    """
    st.title("🚀 Análisis Avanzado de Finanzas")
    
    # Análisis básico
    total_ingresos = df[df['Tipo'] == 'Ingreso']['Monto'].sum()
    total_gastos = df[df['Tipo'] == 'Gasto']['Monto'].sum()
    balance = total_ingresos - total_gastos
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Ingresos", f"${total_ingresos:.2f}")
    with col2:
        st.metric("Total Gastos", f"${total_gastos:.2f}")
    with col3:
        st.metric("Balance", f"${balance:.2f}", 
                  delta_color="inverse",
                  delta=f"{(balance/total_ingresos)*100:.2f}%" if total_ingresos > 0 else "N/A")
    
    # Pestañas de análisis
    tab1, tab2, tab3, tab4 = st.tabs([
        "Resumen Financiero", 
        "Análisis PCA", 
        "Correlaciones", 
        "Análisis Avanzado"
    ])
    
    with tab1:
        # Gráfico de Ingresos vs Gastos
        plt.figure(figsize=(10, 6))
        plt.bar(['Ingresos', 'Gastos'], [total_ingresos, total_gastos], color=['green', 'red'])
        plt.title('Comparación de Ingresos y Gastos')
        plt.ylabel('Monto ($)')
        st.pyplot(plt)
    
    with tab2:
        # Análisis de Componentes Principales
        resultados_pca = realizar_pca_finanzas(df)
        if resultados_pca:
            insights_pca = interpretar_pca(resultados_pca)
            for insight in insights_pca:
                st.write(f"- {insight}")
            st.pyplot(resultados_pca['grafico_varianza'])
    
    with tab3:
        # Análisis de Correlación
        analisis_correlacion_financiera(df)
    
    with tab4:
        # Análisis Avanzados
        transformaciones_lineales(df)
        analisis_vectorial_finanzas(df)

def registrar_transaccion(tipo):
    """
    Función para registrar transacciones manualmente
    """
    st.title(f"Registrar {tipo}")

    # Campos comunes para ingreso y gasto
    descripcion = st.text_input(f"Descripción del {tipo.lower()}")
    monto = st.number_input("Monto en pesos mexicanos", min_value=0.0)
    pago = st.selectbox("Forma de pago", formas_pago)
    fecha = st.date_input("Fecha de transacción", datetime.today())

    # Si es un gasto, añadir valoración de necesidad
    if tipo == "Gasto":
        valoracion = st.slider(
            "¿Qué tan necesario fue este gasto?", 
            min_value=1, 
            max_value=6, 
            step=1
        )
        st.markdown("""
            <style>
                .stSlider + .stText {
                    font-size: 14px;
                    color: #333;
                    font-style: italic;
                }
            </style>
            <p style="font-size: 14px; color: #333; font-style: italic;">1 = Totalmente innecesario, 6 = Totalmente necesario</p>
        """, unsafe_allow_html=True)

    if st.button(f"Registrar {tipo}"):
        # Convertir la fecha en formato adecuado
        fecha_str = fecha.strftime('%Y-%m-%d')

        if tipo == "Ingreso":
            st.write(f"Ingreso registrado: Descripción: {descripcion}, Monto: {monto}, Forma de pago: {pago}, Fecha: {fecha_str}")
        elif tipo == "Gasto":
            st.write(f"Gasto registrado: Descripción: {descripcion}, Monto: {monto}, Forma de pago: {pago}, Fecha: {fecha_str}, Valoración: {valoracion}")

def cargar_csv():
    """
    Función para cargar archivos CSV
    """
    archivo = st.file_uploader("Cargar archivo CSV", type=["csv"])
    if archivo:
        try:
            df = pd.read_csv(archivo)
            st.write("Datos cargados exitosamente:")
            st.dataframe(df)
            
            # Mostrar análisis gráfico e insights
            mostrar_analisis(df)
            
        except Exception as e:
            st.error(f"Error al cargar el archivo: {e}")

def mostrar_ejemplo_csv():
    """
    Mostrar ejemplo de formato CSV
    """
    ejemplo = pd.DataFrame({
        "Descripción": ["Ingreso Trabajo", "Compra Supermercado", "Ingreso Extra", "Pago Servicios"],
        "Monto": [1000, 200, 1500, 100],
        "Forma de pago": ["transferencia", "efectivo", "depósito", "efectivo"],
        "Fecha de transacción": ["2024-12-16", "2024-12-16", "2024-12-17", "2024-12-17"],
        "Tipo": ["Ingreso", "Gasto", "Ingreso", "Gasto"]
    })
    st.write("Ejemplo de formato CSV para carga correcta:")
    st.dataframe(ejemplo)
    
    st.download_button(
        label="Descargar archivo ejemplo",
        data=ejemplo.to_csv(index=False),
        file_name="ejemplo_finanzas_personales.csv",
        mime="text/csv"
    )

def main():
    """
    Función principal de la aplicación
    """
    st.title("💰 Gestor de Finanzas Personales")
    
    # Inyectar CSS personalizado
    st.markdown("""
        <style>
            .stButton > button {
                background-color: #6a1b9a;
                color: white;
                border-radius: 50px;
            }
            .stMetric {
                background-color: #f0f2f6;
                padding: 10px;
                border-radius: 10px;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Menú de navegación
    menu = st.sidebar.radio("Selecciona una opción", 
                             ["Registro Manual", "Carga CSV", "Ejemplo de Datos"])
    
    if menu == "Registro Manual":
        transaccion = st.sidebar.radio("¿Qué deseas registrar?", ["Ingreso", "Gasto"])
        registrar_transaccion(transaccion)
    
    elif menu == "Carga CSV":
        cargar_csv()
    
    elif menu == "Ejemplo de Datos":
        mostrar_ejemplo_csv()

# Ejecutar la aplicación
if __name__ == "__main__":
    main()
