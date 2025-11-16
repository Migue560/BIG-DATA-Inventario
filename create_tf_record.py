import tensorflow as tf
import os
from PIL import Image
import io
import xml.etree.ElementTree as ET
from tqdm import tqdm

# --- CONFIGURACIÓN DE RUTAS Y CLASES ---
# Estas rutas son relativas al directorio de ejecución (inventario/)
IMAGE_DIR = 'images'
ANNOTATION_DIR = 'annotations'
OUTPUT_FILE = 'model/train.record'

# La lista de clases DEBE coincidir con las etiquetas que usaste en LabelImg.
# El orden aquí define los IDs de clase (ID 1 = 'CPU', ID 2 = 'Mesa', etc.)
CLASS_NAMES = ['CPU', 'Mesa', 'Mouse', 'Pantalla', 'Silla', 'Teclado']
# El ID del modelo siempre comienza en 1.

# --- FUNCIONES DE AYUDA ---

def int64_feature(value):
    """Retorna un tf.train.Feature que contiene un valor int64_list."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
    """Retorna un tf.train.Feature que contiene un valor bytes_list."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def float_feature(value):
    """Retorna un tf.train.Feature que contiene un valor float_list."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def create_tf_example(img_path, xml_path):
    """
    Convierte una imagen y su anotación XML en un objeto tf.train.Example.
    """
    # 1. Cargar imagen y metadatos
    try:
        with tf.io.gfile.GFile(img_path, 'rb') as fid:
            encoded_data = fid.read()
        
        encoded_io = io.BytesIO(encoded_data)
        image = Image.open(encoded_io)
    except Exception as e:
        print(f"Error cargando imagen {img_path}: {e}")
        return None

    width, height = image.size
    filename = os.path.basename(img_path).encode('utf8')
    image_format = b'png' if img_path.lower().endswith('.png') else b'jpg'

    # 2. Parsear el archivo XML con manejo de errores de codificación (tu solicitud)
    xmins, xmaxs, ymins, ymaxs = [], [], [], []
    classes_text, classes = [], []
    
    try:
        # Intento 1: Parseo normal (la codificación por defecto)
        tree = ET.parse(xml_path) 
        root = tree.getroot()
    except ET.ParseError:
        # Intento 2: Si el parseo falla, intenta con codificación UTF-8
        try:
            tree = ET.parse(xml_path, parser=ET.XMLParser(encoding='utf-8'))
            root = tree.getroot()
        except Exception as e:
            print(f"Error CRÍTICO cargando/parseando XML {xml_path} después de UTF-8: {e}")
            return None
    except Exception as e:
        print(f"Error CRÍTICO cargando/parseando XML {xml_path}: {e}")
        return None
        
    for obj in root.iter('object'):
        name = obj.find('name').text
        
        # Ignorar objetos que no están en nuestra lista de clases
        if name not in CLASS_NAMES:
            continue

        try:
            # TF Object Detection API usa IDs base 1, por eso sumamos 1
            cls_id = CLASS_NAMES.index(name) + 1 
        except ValueError:
            print(f"Clase '{name}' no encontrada en CLASS_NAMES. Omitiendo.")
            continue
            
        # Obtener y normalizar las coordenadas de la caja
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text) / width
        ymin = float(bbox.find('ymin').text) / height
        xmax = float(bbox.find('xmax').text) / width
        ymax = float(bbox.find('ymax').text) / height

        # Agregar datos a las listas
        xmins.append(xmin)
        xmaxs.append(xmax)
        ymins.append(ymin)
        ymaxs.append(ymax)
        classes_text.append(name.encode('utf8'))
        classes.append(cls_id) 

    # 3. Crear el objeto tf.train.Example
    if not xmins: # Si no hay objetos detectados válidos en el XML
        return None 
        
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/filename': bytes_feature(filename),
        'image/source_id': bytes_feature(filename),
        'image/encoded': bytes_feature(encoded_data),
        'image/format': bytes_feature(image_format),
        'image/object/bbox/xmin': float_feature(xmins),
        'image/object/bbox/xmax': float_feature(xmaxs),
        'image/object/bbox/ymin': float_feature(ymins),
        'image/object/bbox/ymax': float_feature(ymaxs),
        'image/object/class/text': bytes_feature(classes_text),
        'image/object/class/label': int64_feature(classes),
    }))
    return tf_example

# --- FUNCIÓN PRINCIPAL DE EJECUCIÓN ---
def main():
    
    # LÍNEAS DE DEPURACIÓN PARA VERIFICAR RUTAS
    print("--- INICIANDO DEPURACIÓN DE RUTAS ---")
    print(f"Directorio de trabajo actual (CWD): {os.getcwd()}")
    print(f"Buscando carpeta de imágenes en: {os.path.join(os.getcwd(), IMAGE_DIR)}")
    print(f"Buscando carpeta de anotaciones en: {os.path.join(os.getcwd(), ANNOTATION_DIR)}")
    print("-------------------------------------")
    
    if not os.path.exists(IMAGE_DIR) or not os.path.exists(ANNOTATION_DIR):
        print(f"ERROR: No se encuentran los directorios '{IMAGE_DIR}' o '{ANNOTATION_DIR}'. Asegúrate de que los nombres sean exactos ('images' y 'annotations').")
        return

    # Crear el directorio 'model/' si no existe
    if not os.path.exists('model'):
        os.makedirs('model')
        
    writer = tf.io.TFRecordWriter(OUTPUT_FILE)
    
    # Obtener la lista de imágenes a procesar
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg'))]
    
    print(f"Iniciando la creación del TFRecord con {len(image_files)} imágenes...")
    
    # Usamos tqdm para la barra de progreso
    for img_file in tqdm(image_files, desc="Procesando imágenes"):
        img_path = os.path.join(IMAGE_DIR, img_file)
        
        # Determinar la ruta del XML asociado
        base_name = os.path.splitext(img_file)[0]
        xml_file = base_name + '.xml'
        xml_path = os.path.join(ANNOTATION_DIR, xml_file)
        
        if not os.path.exists(xml_path):
            print(f"\nAdvertencia: Archivo XML no encontrado para {img_file}. Revisar la coincidencia de nombres en 'images' y 'annotations'. Omitiendo.")
            continue
            
        tf_example = create_tf_example(img_path, xml_path)
        
        if tf_example:
            writer.write(tf_example.SerializeToString())

    writer.close()
    print(f"\n✅ ¡TFRecord creado con éxito en: {OUTPUT_FILE}!")

if __name__ == '__main__':
    main()