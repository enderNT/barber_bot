# Arquitectura de estado corto y memoria duradera

La implementacion actual usa `LangGraph` para estado corto por `conversation_id`, `Postgres` para memoria duradera filtrada y checkpoints persistentes, y un router de estado con guards deterministas mas clasificador LLM.

## Estado principal

- estado vivo del hilo en `LangGraph`
- memoria duradera consultada por `contact_id`
- recuperacion vectorial opcional con `Qdrant`
- contexto estatico cargado desde `config/barbershop.json`

## Rutas del flujo

- el router decide `conversation`, `rag` o `booking` usando estado compacto
- los nodos actualizan `active_goal`, `stage`, `pending_question`, `booking_details`, `last_tool_result` y el `conversation_summary`
- `booking_details` se vacia cuando la conversacion deja de depender de cita

## Heuristicas

- saludos y turnos triviales permanecen en `conversation`
- preguntas sobre horarios, precios, servicios, pagos o barberos pasan a `rag`
- deseos de agendar y respuestas cortas de seguimiento permanecen en `booking`
