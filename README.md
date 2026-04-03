# Barbershop Assistant

Backend en Python para una barberia que recibe mensajes por webhook de Chatwoot, usa un proveedor `LLM` configurable para generacion y clasificacion de estado, orquesta con `LangGraph`, mantiene continuidad conversacional corta con checkpoints persistentes en `Postgres`, memoria duradera semantica en `Postgres` y prepara recuperacion RAG con `Qdrant`.

## Componentes

- `FastAPI` para el webhook `POST`.
- `LangGraph` para el flujo conversacional con estado corto por `conversation_id`.
- Un proveedor `LLM` configurable como backend remoto de generacion, resumen y clasificacion de estado.
- `AsyncPostgresSaver` para persistir checkpoints del hilo.
- `AsyncPostgresStore` para memoria duradera filtrada y busqueda semantica por `contact_id`.
- `Qdrant` como vector store para el nodo RAG, con modo de simulacion habilitado por defecto.
- Configuracion local estatica para servicios, horarios, barberos y politicas, cargada solo cuando la rama de RAG o booking la necesita.

## Setup local

1. Crear y activar entorno virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Instalar dependencias:

```bash
pip install -e ".[dev]"
```

3. Preparar variables de entorno:

```bash
cp .env.example .env
```

4. Ajustar `config/barbershop.json` con los datos reales de la barberia. Ese archivo alimenta el contexto de RAG y la extraccion de intencion de cita, no el router ni la conversacion general.

5. Exportar la configuracion del proveedor LLM en tu entorno:

```bash
export LLM_PROVIDER="openai_compatible"
export LLM_API_KEY="..."
export LLM_MODEL="gpt-5-mini"
```

Si usas un endpoint compatible con OpenAI, tambien puedes definir `LLM_BASE_URL`.

6. Configurar Postgres para memoria y checkpoints:

```bash
export MEMORY_BACKEND="postgres"
export CHECKPOINT_BACKEND="postgres"
export POSTGRES_DSN="postgresql://postgres:postgres@localhost:5432/barbershop_assistant"
```

La primera vez, la app crea las tablas necesarias mediante `store.setup()` y `checkpointer.setup()`. Si prefieres desarrollo sin persistencia, puedes usar `MEMORY_BACKEND=in_memory` y `CHECKPOINT_BACKEND=memory`.

7. Ejecutar la API:

```bash
uvicorn app.main:create_app --factory --reload
```

8. Si necesitas exponer el webhook localmente con `ngrok`, puedes usar:

```bash
make ngrok
make webhook-url
```

Opcionalmente define `NGROK_AUTHTOKEN` y `NGROK_DOMAIN` en `.env` si quieres autenticar el agente o fijar una URL.

9. Si vas a usar Qdrant real, configurar `QDRANT_ENABLED=true`, `QDRANT_SIMULATE=false` y apuntar `QDRANT_BASE_URL` al cluster o instancia local. Si no, el flujo RAG usa simulacion controlada y sigue funcionando.

## Testing

Ejecuta la suite con:

```bash
pytest -q
```

El repo ahora declara `pythonpath = ["."]`, por lo que `app` queda importable sin ajustes manuales adicionales.

## Flujo

1. Chatwoot envia un `POST` al webhook.
2. La API responde inmediatamente con un acuse.
3. En segundo plano se recuperan pocas memorias relevantes desde `Postgres` por `contact_id`, mientras el estado corto del hilo se lee y persiste por `conversation_id`.
4. Un router de estado aplica guards deterministas y, si hace falta, un clasificador LLM para decidir entre conversacion general, RAG o booking.
5. Solo si la rama es `rag` o `booking`, se carga `config/barbershop.json` para construir el contexto completo de la barberia.
6. La respuesta se envia por la API de Chatwoot si esta habilitada; si no, queda registrada en logs.

## Git

El repositorio se inicializa localmente, pero no se hace commit automatico ni se versiona nada por defecto.
