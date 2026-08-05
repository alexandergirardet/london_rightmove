

# Sistema de Predicción de Alquileres de Rightmove

En los últimos meses, he profundizado mi experiencia en la creación de sistemas de aprendizaje automático (ML) a través del estudio y la aplicación exhaustiva de tres áreas fundamentales: Ingeniería de Datos, MLOps e Ingeniería de ML, según lo estructurado por los cursos de Data Talk Club. Mi proyecto, el Sistema de Predicción de Alquileres de Rightmove, encapsula este recorrido, demostrando una aplicación cohesiva de estas habilidades.

La esencia de este proyecto radica en su arquitectura integral, diseñada para predecir precios de alquiler con precisión. Integra:

1. **Ingeniería de Datos** a través de un raspador web asíncrono y pipelines de ingestión por lotes, lo que permite una extracción y preprocesamiento de datos eficientes.
2. **Ingeniería de ML** con un enfoque en el entrenamiento de modelos y la ingeniería de características, incluido el desarrollo de una innovadora "Walk Score".
3. **MLOps** mediante la implementación de prácticas de monitoreo para garantizar la fiabilidad y el rendimiento del sistema a lo largo del tiempo.

### **Componentes del Proyecto**

1. **Pipeline de Extracción y Procesamiento de Datos**: Automatizado para manejar la extracción, limpieza y preparación de datos a gran escala.
2. **Pipeline de Entrenamiento de ML**: Diseñado para la experimentación y el entrenamiento iterativos, aprovechando un modelo RandomForest entre otros, para identificar el método de predicción más efectivo.
3. **Pipeline de Monitoreo de MLOps**: Garantiza que el rendimiento del modelo se mantenga óptimo a través del monitoreo continuo de la deriva de datos y otros problemas potenciales.
4. **API de Servicio de Modelos**: Utiliza FastAPI para una implementación eficiente del modelo, permitiendo predicciones en tiempo real.
5. **Panel de Visualización**: Construido con Streamlit y Grafana, ofrece visualizaciones de datos perspicaces y paneles de monitoreo para rastrear el rendimiento del sistema y la calidad de los datos.

### **Infraestructura e Implementación**

Mi enfoque combina principios de DevOps e ingeniería de software, empleando Terraform para la gestión de infraestructura y Docker Compose para la contenerización, en las plataformas AWS y GCP. Esta estrategia de doble nube no solo aprovecha los puntos fuertes de ambos servicios, sino que también optimiza los costos a través de sus opciones de nivel gratuito.

### **Implementación de ML y MLOps**

El proyecto muestra mi experiencia en ML y MLOps a través del desarrollo de un modelo RandomForest, mejorado por una característica única, la Walk Score, para mejorar la precisión predictiva. MLFlow sirve como columna vertebral para el seguimiento de experimentos y el registro de modelos, facilitando la evolución y el servicio del modelo.

### **Extracción y Procesamiento de Datos**

![Data Extraction Pipeline](/static/images/Processing_pipeline_rightmove.png)

Al elegir Rightmove, un sitio líder de listados de propiedades en el Reino Unido, como fuente de datos, desarrollé un spider Scrapy desplegado en un servidor Scrapyd. Esta configuración mejora el control sobre las actividades de raspado e integra sin problemas con Airflow para la orquestación, garantizando un uso ético de los datos y el cumplimiento de las mejores prácticas.

El almacenamiento de datos se gestiona a través de PostgreSQL y MongoDB, que admiten datos estructurados y no estructurados, respectivamente. Esta configuración no solo facilita una gestión eficiente de los datos, sino que también integra un trabajo personalizado de Beam para calcular la Walk Score y mejorar la entrada del modelo.

### **Entrenamiento de ML con MLFlow**

![ML Training pipeline](/static/images/model_training_pipeline.png)

Para el componente de entrenamiento de ML, MLFlow desempeñó un papel crítico como centro principal para el seguimiento de experimentos, el control de versiones de modelos y el servicio. Esta herramienta permitió un enfoque sistemático para gestionar el ciclo de vida de los modelos de aprendizaje automático. Aquí se explica cómo se integró en el flujo de trabajo:

- **Seguimiento de Experimentos**: Cada ejecución de entrenamiento, junto con sus parámetros, métricas y resultados, se registró en MLFlow. Esto facilitó un análisis exhaustivo de cada experimento, permitiendo una rápida iteración sobre los modelos para encontrar los que tenían mejor rendimiento según las métricas de Error Cuadrático Medio (RMSE).
- **Registro de Modelos**: Los modelos más prometedores, particularmente el modelo RandomForest que superó a otros, incluido XGBoost, se registraron en el registro de modelos de MLFlow. Este registro actuó como un repositorio, facilitando el control de versiones, el almacenamiento y el acceso a los modelos para su implementación.
- **Servicio de Modelos**: MLFlow también optimizó el proceso de implementación. El componente de servicio obtenía la versión del modelo más reciente y efectiva directamente desde el registro, asegurando que el servicio de predicción siempre utilizara el mejor modelo disponible.

El uso de MLFlow no solo aportó organización y eficiencia al proceso de entrenamiento de modelos, sino que también garantizó transparencia y reproducibilidad, aspectos esenciales para la colaboración y la mejora continua en proyectos de ML.

## **DevOps y Monitoreo del Raspador**

El Sistema de Predicción de Alquileres de Rightmove emplea un enfoque centrado para monitorear sus operaciones de raspado web, aprovechando Grafana y PostgreSQL para una supervisión ágil y efectiva.

### **Marco de Monitoreo**

**Panel de Grafana**: Proporciona visualización en tiempo real de métricas críticas como tasas de éxito, conteo de errores y tiempos de respuesta. Este panel permite identificar rápidamente problemas de rendimiento o errores en el proceso de raspado web.

**PostgreSQL**: Actúa como la columna vertebral de almacenamiento para registrar métricas detalladas de cada sesión de raspado. Esto incluye marcas de tiempo, conteo de registros extraídos y registros de errores, ofreciendo una visión integral para el análisis y la solución de problemas.

### **Objetivos Clave**

- **Eficiencia y Gestión de Errores**: El monitoreo garantiza que el raspador funcione de manera eficiente, con una respuesta rápida ante cualquier error o cuello de botella.
- **Cumplimiento y Limitación de Tasa**: Mantiene las actividades de raspado dentro de límites éticos y legales al rastrear las tasas de solicitud y el cumplimiento de las políticas del sitio.

### **Integración con DevOps**

La configuración se integra sin problemas con nuestras prácticas de DevOps, con alertas de Grafana configuradas para desencadenar acciones automatizadas o notificaciones para atención inmediata, garantizando la robustez y fiabilidad del sistema.

#### Monitoreo del Sistema
![Extraction Monitoring](/static/images/scrapy_monitoring.png)

Monitoreo del sistema de sesiones de Scrapy

## **MLOps**

![MLOps Diagram](/static/images/mlops_pipeline.png)

Comprender y mitigar concept drift y data drift son aspectos críticos para mantener el rendimiento de los modelos de ML en producción. Así es como se abordaron estos desafíos:

- **Concept Drift**: Esto ocurre cuando las propiedades estadísticas de la variable objetivo, que el modelo intenta predecir, cambian con el tiempo. Esto puede degradar el rendimiento del modelo porque los patrones que el modelo aprendió durante el entrenamiento ya pueden no aplicarse. Para detectar concept drift, el pipeline de monitoreo empleó pruebas estadísticas y comparaciones entre las predicciones y los resultados reales a lo largo del tiempo. Cuando se detectó una deriva significativa, se desencadenó un flujo de trabajo de reentrenamiento del modelo, incorporando nuevos datos para adaptar el modelo a la realidad actual.
- **Data Drift**: Data drift se refiere a los cambios en la distribución de los datos de entrada. Es crucial monitorearla porque, incluso si la distribución de la variable objetivo permanece igual, los cambios en los datos de entrada pueden provocar un mal rendimiento del modelo. El proyecto utilizó Evidently para monitorear las distribuciones de las características clave, comparando los datos entrantes con una línea base histórica (el golden dataset). Se configuraron alertas para notificar cuando data drift superaba umbrales predefinidos, lo que impulsaba una evaluación para determinar si era necesario reentrenar el modelo o ajustar los pasos de preprocesamiento de datos.

#### Monitoreo del Modelo de ML
![Model Monitoring](/static/images/model_monitoring.png)

Monitoreo de MLOps de Data y Concept Drift

### Abordando el cambio

Grafana permite acciones automatizadas. En el caso de que el rendimiento de predicción de nuestro modelo caiga por debajo de un cierto umbral, desencadenaremos un reentrenamiento automático del modelo con nuevos datos que incluyan los nuevos patrones. Esto debería garantizar que nuestro modelo se mantenga actualizado de manera automatizada.
