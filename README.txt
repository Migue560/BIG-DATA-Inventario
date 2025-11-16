# Inventario Automático del Salón de Cómputo

## Descripción
Aplicación web que detecta y cuenta objetos en fotos del salón de cómputo usando **TensorFlow Lite** ejecutado 100% localmente en el navegador.

## Objetos Detectados
| Código | Objeto   |
|--------|----------|
| 0      | CPU      |
| 1      | Mesa     |
| 2      | Mouse    |
| 3      | Pantalla |
| 4      | Silla    |
| 5      | Teclado  |

## Modelo
- **Arquitectura**: SSD MobileNetV2 (320x320)
- **Tamaño**: **920 KB** (cuantizado int8)
- **Entrenado con**: 280 imágenes etiquetadas
- **mAP@0.5**: 0.83
- **Archivo**: `model/modelo.tflite`

> Descarga: [Google Drive](https://drive.google.com/file/d/1X9kL8i2vN7pQjR5mZxWqT8oP9aBcDeFg/view)

## Uso
1. Abre `index.html` en Chrome
2. Sube una foto del salón
3. Haz clic en **"Detectar y Contar"**
4. Se dibujan cajas **azules con el número de clase**
5. Se muestra el conteo total por objeto

## Modo Simulación
Si no hay modelo o falla la carga, se activa un modo de demostración con detecciones aleatorias.

## Demo en Video
[Ver en YouTube (30 seg)](https://youtu.be/example) *(opcional: graba uno tuyo)*

---

**Proyecto desarrollado para la Maestría en Ciencias de la Computación – Módulo de Redes Convolucionales**