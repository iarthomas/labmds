Conclusiones
¿Cómo mejoró el desarrollo del proyecto al utilizar herramientas de tracking y despliegue?

El proyecto mostró lo valioso que es trabajar con un pipeline orquestado y reproducible. Airflow permitió transformar un conjunto de scripts aislados en un flujo estructurado, claro y fácil de depurar. Ver cada paso separado —ingesta, dataset supervisado, split temporal, entrenamiento y optimización— ayudó a entender mejor la lógica del problema y a pensar en un sistema continuo más que en un experimento puntual.
La trazabilidad de ejecuciones, fechas y modelos guardados por día aportó orden, historia y una base concreta para escalar el sistema.

¿Qué aspectos del despliegue con FastAPI/Gradio fueron más desafiantes o interesantes?

Integrar el backend con FastAPI fue una experiencia desafiante y formativa. Traducir lo que hace el pipeline a una API robusta exigió precisión en el manejo de formatos, carga del modelo, validación y control de errores. El punto más crítico fue alinear exactamente las features que el frontend enviaba con lo que esperaba el modelo entrenado, especialmente por el uso de ColumnTransformer y features generadas automáticamente.

Gradio, por otro lado, permitió construir una interfaz clara en muy poco tiempo. El reto estuvo en pasar de un “demo” simple a algo con mejor estructura visual, explicaciones y una experiencia más cercana a un producto final.

¿Cómo aporta Airflow a la robustez y escalabilidad del pipeline?

Airflow fue probablemente la pieza que más aportó a la sensación de robustez. Su estructura facilita reproducibilidad, orden y claridad. Tener ejecuciones fechadas, artefactos versionados y la posibilidad de reentrenar sin manipulación manual generó una base sólida para pensar en escalabilidad real.
Además, el diseño modular del DAG permite extender el sistema hacia tareas más avanzadas, como validación automática, monitoreo de drift o reentrenamientos periódicos.

¿Qué se podría mejorar en una versión futura del flujo?

Hay varias oportunidades claras para una versión más completa:

Monitoreo y detección de drift, que permitiría decidir automáticamente cuándo reentrenar.

MLflow, para versionar modelos formalmente, comparar ejecuciones y permitir rollback seguro.

CI/CD, para automatizar por completo las actualizaciones del backend y frontend.

Frontend más explicativo, incorporando interpretabilidad o visualizaciones que apoyen la toma de decisiones.

Estas mejoras empujarían el proyecto hacia un producto más autónomo, escalable y mantenible.