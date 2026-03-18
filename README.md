# Clinica Assistant

Backend en Python para una clinica que recibe mensajes por webhook de Chatwoot, usa `OpenAI` para generacion y clasificacion de estado, orquesta con `LangGraph`, mantiene continuidad conversacional corta con estado de hilo y memoria duradera con `mem0`, y prepara recuperacion RAG con `Qdrant`.

## Componentes

- `FastAPI` para el webhook `POST`.
- `LangGraph` para el flujo conversacional con estado corto por `conversation_id`.
- `OpenAI` como proveedor remoto de generacion, resumen y clasificacion de estado.
- `mem0` para memoria duradera filtrada.
- `Qdrant` como vector store para el nodo RAG, con modo de simulacion habilitado por defecto.
- Configuracion local estatica para servicios, horarios, doctores y politicas, cargada solo cuando la rama de RAG o cita la necesita.

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

4. Ajustar `config/clinic.json` con los datos reales de la clinica. Ese archivo alimenta el contexto de RAG y la extraccion de intencion de cita, no el router ni la conversacion general.

5. Exportar credenciales de OpenAI en tu entorno:

```bash
export OPENAI_API_KEY="..."
export OPENAI_MODEL="gpt-5-mini"
```
6. Ejecutar la API:

```bash
uvicorn app.main:create_app --factory --reload
```

7. Si necesitas exponer el webhook localmente con `ngrok`, puedes usar:

```bash
make ngrok
make webhook-url
```

Opcionalmente define `NGROK_AUTHTOKEN` y `NGROK_DOMAIN` en `.env` si quieres autenticar el agente o fijar una URL.

8. Si vas a usar Qdrant real, configurar `QDRANT_ENABLED=true`, `QDRANT_SIMULATE=false` y apuntar `QDRANT_BASE_URL` al cluster o instancia local. Si no, el flujo RAG usa simulacion controlada y sigue funcionando.

## Flujo

1. Chatwoot envia un `POST` al webhook.
2. La API responde inmediatamente con un acuse.
3. En segundo plano se recuperan pocas memorias relevantes de `mem0` y el estado corto del hilo viaja en `LangGraph`.
4. Un router de estado aplica guards deterministas y, si hace falta, un clasificador LLM para decidir entre conversacion general, RAG o cita.
5. Solo si la rama es `rag` o `appointment`, se carga `config/clinic.json` para construir el contexto clinico completo.
6. La respuesta se envia por la API de Chatwoot si esta habilitada; si no, queda registrada en logs.

## Git

El repositorio se inicializa localmente, pero no se hace commit automatico ni se versiona nada por defecto.
