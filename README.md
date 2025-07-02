# Clasificación de Resistencias Eléctricas a partir de Imágenes

Este proyecto consiste en la detección y clasificación automática de resistencias eléctricas a partir de imágenes tomadas desde diferentes ángulos. 
Fue desarrollado como parte de un ejercicio práctico sobre procesamiento de imágenes y visión por computadora.

El procedimiento completo incluye:

- Detección de un rectángulo azul delimitador.
- Corrección de perspectiva (homografía).
- Segmentación de la resistencia.
- Detección de bandas de colores.
- Clasificación del valor de resistencia en ohmios.

## 📁 Estructura del repositorio

```
clasificacion-resistencias/
│
├── ejercicio2.py             # Script principal de detección y clasificación
├── requirements.txt          # Dependencias del proyecto
│
├── /Resistencias/            # Carpeta con imágenes de entrada (R1_a.jpg, R2_a.jpg, ...)
├── /resistencias_out/        # Se genera automáticamente con las imágenes procesadas
```

> 💡 El script genera imágenes corregidas y realiza la lectura de bandas para cada resistencia.

## ▶️ Cómo ejecutar

1. Clonar el repositorio:
```bash
git clone https://github.com/ValentinaBalverdi/Clasificacion-resistencias-electricas.git
cd clasificacion-resistencias-electricas
```
2. Crear un entorno virtual (opcional, pero recomendado):
```bash
# Para Linux/macOS
python3 -m venv venv
source venv/bin/activate

# Para Windows
python -m venv venv
venv\\Scripts\\activate
```
3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

4. Ejecutar el script:
```bash
python main.py
```

Esto procesará las imágenes dentro de la carpeta `Resistencias` y guardará las correcciones en `resistencias_out`. 
Además, mostrará en consola los colores detectados y el valor de la resistencia.

## 🧠 Ejemplo de salida

```
Resistencia R1_a_out.jpg:
  Banda 1: Rojo
  Banda 2: Violeta
  Banda 3: Marron
  Valor: 270 Ω
```
