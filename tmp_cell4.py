# Celda 3: Generación de Tareas y Envío del Batch a OpenAI

import json
import io
import datetime
from openai import OpenAI # Re-importar o asegurarse de que el cliente esté disponible
from dotenv import load_dotenv

load_dotenv()



# --- Verificar si las variables necesarias existen (creadas en celdas anteriores) ---
if 'df_filtrado' not in globals() or df_filtrado.empty:
    print("Error CRÍTICO: DataFrame 'df_filtrado' no disponible o vacío. Ejecuta Celda 1 y Celda 2.")
    raise NameError("'df_filtrado' no disponible o vacío.")

if 'final_system_prompt' not in globals() or not final_system_prompt:
    print("Error CRÍTICO: 'final_system_prompt' no disponible. Ejecuta Celda 2.")
    raise NameError("'final_system_prompt' no disponible.")

# --- Inicializar cliente OpenAI (si no se hizo globalmente o si el kernel se reinició) ---
# Asumiendo que las variables de entorno ya fueron cargadas por load_dotenv() en Celda 1
try:
    client = OpenAI()
    print("Cliente OpenAI inicializado/confirmado.")
except Exception as e:
    print(f"Error CRÍTICO al inicializar el cliente de OpenAI: {e}")
    raise

# --- Generación de Tareas para el Batch ---
print("\nGenerando lista de tareas para el batch...")
batch_tasks_list = []
for index_original, row_data in df_filtrado.iterrows(): # `index_original` es el índice de df_filtrado
    noticia_description = str(row_data['body']) if pd.notna(row_data['body']) else ""
    
    # No es necesario volver a filtrar por descripción vacía aquí si df_filtrado ya lo hizo (word_count > 0)
    # pero una comprobación extra no hace daño si la lógica de filtrado cambia.
    if not noticia_description.strip(): 
        # print(f"Saltando índice {index_original} por descripción vacía (inesperado si df_filtrado es correcto).")
        continue

    task_item = {
        "custom_id": f"task-{index_original}", # Usa el índice original de la noticia
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4.1-mini", # Asegúrate que este es el modelo deseado
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": final_system_prompt},
                {"role": "user", "content": noticia_description}
            ],
        }
    }
    batch_tasks_list.append(task_item)

print(f"Número total de tareas generadas para el batch: {len(batch_tasks_list)}")

# --- Creación del Archivo Batch en Memoria y Envío ---
if not batch_tasks_list:
    print("No se generaron tareas. El batch no será enviado.")
else:
    # Confirmación antes de enviar (opcional, pero recomendado)
    user_confirmation_envio = input(f"Se han generado {len(batch_tasks_list)} tareas. ¿Deseas proceder con el envío del batch a OpenAI? (s/N): ")
    if user_confirmation_envio.lower() != 's':
        print("Envío del batch cancelado por el usuario.")
    else:
        print("\nProcediendo con la creación del archivo batch y envío a OpenAI...")
        jsonl_content_list_for_batch = [json.dumps(obj) for obj in batch_tasks_list]
        jsonl_content_str_for_batch = '\n'.join(jsonl_content_list_for_batch)

        batch_file_in_memory = io.BytesIO(jsonl_content_str_for_batch.encode('utf-8'))
        batch_file_in_memory.name = f"batch_input_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

        print(f"Creando archivo batch '{batch_file_in_memory.name}' en OpenAI...")
        try:
            openai_batch_file_object = client.files.create(
                file=batch_file_in_memory,
                purpose="batch"
            )
            print(f"Archivo batch creado en OpenAI con ID: {openai_batch_file_object.id}")

            print("Enviando job batch a OpenAI...")
            openai_batch_job_object = client.batches.create(
                input_file_id=openai_batch_file_object.id,
                endpoint="/v1/chat/completions",
                completion_window="24h" # Tiempo para que OpenAI complete el batch
            )
            print(f"Job batch enviado con ID: {openai_batch_job_object.id}")
            print("Puedes monitorear el estado del batch en el dashboard de OpenAI o usando la API.")
            print(f"Una vez completado, descarga el archivo de resultados JSONL (output).")
            print(f"Asegúrate de tener 'df_filtrado' (con {len(df_filtrado)} filas) disponible en memoria si Celda 4 lo necesita directamente.")

        except Exception as e:
            print(f"Error CRÍTICO durante la creación o envío del batch a OpenAI: {e}")
            # Considera `raise e` si el error debe detener todo flujo posterior