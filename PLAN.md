# Plan de Refactor: Estado Corto con LangGraph y Memoria Duradera con mem0

## Resumen
Migrar el flujo actual de `último mensaje + semantic-router + mem0` a una arquitectura híbrida donde:

- **LangGraph** maneja el **estado corto del hilo/conversación**
- **mem0** se queda como **memoria duradera**
- **Aurelio Labs / semantic-router** se elimina por completo
- La decisión de `next_node` pasa a ser **reglas deterministas + clasificador LLM sobre estado compacto**
- **Qdrant** sigue solo para RAG/documentos

Defaults cerrados para esta versión:
- **Corto plazo**: estado vivo en LangGraph con checkpointer simple de hilo
- **Largo plazo**: mem0 como memoria externa filtrada
- **Routing**: guards primero, luego LLM de estado
- **Sin LangGraph Store** en esta fase

## Cambios de implementación
### 1. Estado corto del hilo con LangGraph
- Convertir `conversation_id` en el identificador oficial del hilo.
- Dejar de tratar cada webhook como flujo completamente aislado.
- Expandir el estado del grafo para incluir, como mínimo:
  - `active_goal`
  - `stage`
  - `pending_action`
  - `pending_question`
  - `appointment_slots`
  - `conversation_summary`
  - `last_tool_result`
  - `last_user_message`
  - `last_assistant_message`
- El estado corto debe vivir solo en LangGraph y representar la conversación activa, no recuerdos duraderos.

### 2. Memoria duradera con mem0
- Mantener mem0 como sistema de memoria externa.
- Separar conceptualmente lo que se guarda en mem0 en dos tipos:
  - **perfil estable**: preferencias, restricciones, estilo, datos persistentes
  - **episodios útiles**: recuerdos relevantes que sí aporten en conversaciones futuras
- No guardar en mem0:
  - saludos
  - confirmaciones triviales
  - todo el historial bruto
  - tool outputs completos
- Recuperar de mem0 solo pocas memorias relevantes por turno y pasarlas ya resumidas al paquete de routing.

### 3. Sustitución total de Aurelio Labs
- Eliminar `semantic-router` de dependencias, settings, tests, docs y observabilidad.
- Sustituir el router actual por un **State Routing Service**.
- El nuevo servicio debe decidir con un `routing_packet` compacto que incluya:
  - mensaje actual
  - resumen corto del hilo
  - objetivo activo
  - etapa actual
  - slots activos
  - último resultado útil
  - memorias relevantes de mem0
- Orden fijo de decisión:
  1. **Guard rails deterministas**
  2. **Clasificador LLM de estado**
  3. **Retrieval o lookup adicional solo si se necesita**
- El clasificador debe devolver salida estructurada con:
  - `next_node`
  - `intent`
  - `confidence`
  - `needs_retrieval`
  - `state_update`

### 4. Lógica de nodos y actualización de estado
- **conversation**:
  - responde sin retrieval
  - actualiza resumen, objetivo y etapa cuando cambie el tema
- **rag**:
  - se ejecuta solo si `next_node=rag` o si el clasificador marca `needs_retrieval=true`
  - guarda solo un resumen corto del resultado en `last_tool_result`
  - no deja contexto viejo colgando en el estado
- **appointment**:
  - captura slots por turnos
  - mantiene `pending_question` si falta información
  - completa o limpia `appointment_slots` según avance el flujo
- Cada nodo actualiza **estado corto** de forma determinista.
- La escritura a **mem0** ocurre solo cuando un hecho merece memoria duradera.

### 5. Summary y limpieza
- Añadir un paso de **summary incremental** usando el LLM actual.
- El resumen no se recalcula en todos los turnos; se refresca cuando:
  - cambie el objetivo activo
  - crezca demasiado el estado reciente
  - termine un subflujo importante
- Limpiar explícitamente:
  - `last_tool_result` cuando ya no aplique
  - `pending_action` cuando se resuelva
  - slots temporales cuando la cita termine o se cancele

## APIs, tipos y contratos a ajustar
- Mantener `ChatwootWebhook` y la interfaz externa del webhook.
- Reemplazar la decisión actual de intent puro por una decisión de estado que incluya `next_node` y `state_update`.
- Ampliar `GraphState` para soportar el estado conversacional vivo.
- Mantener mem0 como dependencia oficial del proyecto.
- Eliminar settings de `semantic-router`.
- Mantener settings de mem0 y añadir solo lo necesario para control de summary y del clasificador de estado.

## Plan de pruebas
- Una pregunta institucional entra a `rag`, luego el usuario dice “quiero agendar una cita” y el siguiente turno entra a `appointment`.
- Respuestas elípticas como “sí”, “mañana”, “a las 10” permanecen en `appointment` si hay slots pendientes.
- Un flujo activo de cita no se rompe por una respuesta corta.
- RAG no se ejecuta si el estado no lo pide.
- El router ya no depende de similarity del último mensaje.
- mem0 solo guarda recuerdos útiles y no el chat completo.
- El proyecto funciona sin `semantic-router` y sus tests cubren `conversation`, `rag`, `appointment`, continuidad y follow-ups.

## Supuestos y defaults
- `conversation_id` será el hilo oficial para estado corto.
- `contact_id` seguirá siendo la clave principal para mem0.
- LangGraph se usará para **estado corto de conversación**, no para memoria duradera.
- mem0 se usará para **largo plazo**, porque reduce carga operativa comparado con meter LangGraph Store en esta fase.
- OpenAI se mantiene para generación, summary y clasificación de estado.
- Qdrant sigue solo como soporte de RAG.
- La migración se ejecuta en este orden:
  1. introducir estado corto en LangGraph
  2. reemplazar Aurelio por guards + clasificador LLM
  3. adaptar nodos para actualizar estado vivo
  4. afinar summary y limpieza
  5. mantener mem0 como memoria duradera filtrada
