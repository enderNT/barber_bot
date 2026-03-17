# Plan de migracion: vLLM -> OpenAI + semantic-router

> Estado actual: fases 1, 2 y 3 implementadas en el codigo. El router ya usa `semantic-router` con fallback deterministico si falta la dependencia o la key.

## Objetivo

Eliminar la dependencia conceptual y operativa de `vLLM` en este proyecto.

- Generacion de respuestas y extraccion estructurada: `OpenAI` como proveedor remoto.
- Enrutamiento semantico entre nodos de `LangGraph`: `semantic-router` de Aurelio Labs.
- Sin modelos locales ni infraestructura de inferencia propia por ahora.

## Estado actual del proyecto

Hoy existe un acoplamiento fuerte entre routing y generacion:

- `app/services/llm.py` concentra:
  - cliente `VLLMClient`
  - `route_intent`
  - `build_conversation_reply`
  - `build_rag_reply`
  - `extract_appointment_intent`
- `app/graph/workflow.py` depende de un solo servicio para decidir el branch y para generar la salida.
- `app/main.py` construye el flujo inyectando ese servicio unico.
- `app/settings.py` expone solo configuracion orientada a `vLLM`.

Esto funciona, pero mezcla dos responsabilidades distintas:

1. Clasificacion semantica de intencion.
2. Generacion/extraccion con LLM.

## Arquitectura objetivo

Separar por contratos:

- `RouterService`
  - responsable solo de devolver `IntentDecision`
  - implementacion: `SemanticRouterService`
- `GenerationService`
  - responsable de respuesta conversacional, RAG y extraccion de cita
  - implementacion: `OpenAIClinicLLMService`

LangGraph debe depender de ambos servicios por separado.

## Decision de integracion

### 1. Routing con semantic-router

Usar `SemanticRouter` con:

- `OpenAIEncoder`
- `LocalIndex` al inicio
- rutas estaticas

Rutas iniciales propuestas:

- `conversation`
- `rag`
- `appointment_intent`

No recomiendo empezar con dynamic routes para este proyecto.

Motivo:

- El problema actual de LangGraph es escoger nodo, no hacer tool calling desde el router.
- La extraccion de campos de cita ya encaja mejor como tarea de generacion estructurada en OpenAI.
- Dynamic routes agregarian otra capa de LLM dentro del router sin resolver una necesidad inmediata.

### 2. Generacion y extraccion con OpenAI

Usar OpenAI como proveedor remoto para:

- respuesta conversacional
- respuesta RAG
- extraccion de payload de cita

Recomendacion de implementacion:

- SDK oficial `openai`
- endpoint preferente: `Responses API` para generacion futura
- embeddings con `text-embedding-3-small` para el router por costo/latencia

Si se quiere minimizar el cambio inicial, se puede mantener temporalmente una interfaz estilo chat completions dentro del servicio y migrar despues a `Responses API`, pero el contrato interno debe quedar desacoplado de cualquier backend local.

## Diseño propuesto por capas

### Servicios nuevos

- `app/services/router.py`
  - interfaz/base del router
  - `SemanticRouterService`
- `app/services/openai_llm.py`
  - cliente OpenAI
  - servicio de generacion y extraccion

### Refactor del workflow

Cambiar `ClinicWorkflow` para recibir:

- `router_service`
- `generation_service`

Uso esperado:

- `_route` usa `router_service.route_intent(...)`
- `_conversation` usa `generation_service.build_conversation_reply(...)`
- `_rag` usa `generation_service.build_rag_reply(...)`
- `_appointment` usa `generation_service.extract_appointment_intent(...)`

### Configuracion

Reemplazar settings de `vLLM` por settings neutrales o de OpenAI:

- `openai_api_key`
- `openai_base_url` opcional
- `openai_generation_model`
- `openai_router_embedding_model`
- `openai_timeout_seconds`
- `openai_temperature`
- `semantic_router_config_path`
- `semantic_router_auto_train`

`openai_base_url` debe ser opcional para no cerrar la puerta a compatibilidad futura, pero el default debe apuntar a OpenAI y no a infraestructura local.

## Definicion inicial de rutas

### `appointment_intent`

Utterances ejemplo:

- "quiero agendar una cita"
- "necesito una consulta"
- "quiero reservar con un doctor"
- "me ayudas a sacar una cita"
- "busco cita para dermatologia"

### `rag`

Utterances ejemplo:

- "cuales son sus horarios"
- "que servicios ofrecen"
- "cuanto cuesta la consulta"
- "que especialidades tienen"
- "atienden los sabados"

### `conversation`

Utterances ejemplo:

- "hola"
- "gracias"
- "buenos dias"
- "me puedes ayudar"
- "ok perfecto"

Importante:

- `conversation` debe incluir mensajes de cortesia y seguimiento corto.
- `rag` debe capturar preguntas institucionales concretas.
- `appointment_intent` debe capturar deseo operativo de reservar/agendar.

## Estrategia de thresholds

No dejar thresholds por default en produccion.

Plan:

1. Empezar con thresholds por ruta conservadores.
2. Recolectar ejemplos reales anonimizados de Chatwoot.
3. Construir dataset etiquetado:
   - texto
   - ruta esperada
   - casos `None` o ambiguos
4. Usar `fit` y `evaluate` de `semantic-router`.
5. Congelar thresholds en archivo versionado.

Cuando no haya match suficiente:

- fallback a `conversation`
- registrar score y top matches en logs

## Persistencia del router

Guardar definicion del router en archivo versionado.

Propuesta:

- `config/semantic_router_routes.json` o `.yaml`

El codigo puede:

1. construir el router desde Python en desarrollo
2. serializarlo con `to_json` o `to_yaml`
3. cargarlo en runtime con `from_json` o `from_yaml`

Esto permite:

- versionar utterances y thresholds
- revisar cambios por git
- no reentrenar manualmente en cada arranque

## Observabilidad

Cambiar trazas orientadas a `vLLM` por trazas neutrales:

- `2.2 semantic_router`
- `2.2.1 semantic_router_match`
- `2.2.2 semantic_router_fallback`
- `3.a.1 openai_conversation`
- `3.b.1 openai_rag`
- `3.c.1 openai_appointment_extraction`

Registrar:

- route elegida
- similarity score
- threshold aplicado
- top candidates si aplica
- fallback activado o no

## Plan por fases

### Fase 1. Desacople de contratos

- crear interfaces separadas para routing y generacion
- modificar `ClinicWorkflow`
- adaptar tests dobles/fakes

Resultado:

- LangGraph deja de depender de una sola implementacion monolitica

### Fase 2. OpenAI para generacion

- crear cliente OpenAI
- portar prompts existentes
- mantener outputs compatibles con `IntentDecision` y `AppointmentIntentPayload`
- retirar referencias a `VLLMClient`

Resultado:

- el sistema deja de depender de backend local para responder

### Fase 3. semantic-router para branching

- agregar `semantic-router` a dependencias
- definir rutas base
- construir `SemanticRouterService`
- integrar fallback si `RouteChoice.name` es `None`

Resultado:

- el branch de LangGraph se decide por embeddings, no por prompt de clasificacion

### Fase 4. Afinado y persistencia

- crear dataset de evaluacion
- ajustar thresholds con `fit`
- guardar router entrenado/configurado en archivo
- endurecer tests con casos ambiguos

Resultado:

- router estable y trazable

### Fase 5. Limpieza final

- borrar config y naming de `vLLM`
- actualizar README
- revisar logs, nombres y tests

## Riesgos y mitigaciones

### Riesgo: mala separacion entre `rag` y `conversation`

Mitigacion:

- definir utterances negativas/ambiguas
- evaluar con mensajes reales
- usar thresholds por ruta, no uno global ciego

### Riesgo: overfitting del router a pocos ejemplos

Mitigacion:

- incluir ejemplos reales variados
- incluir casos `None`
- revisar accuracy y falsos positivos antes de promover

### Riesgo: depender de OpenAI tanto para embeddings como para generacion

Mitigacion:

- separar contratos
- dejar `openai_base_url` opcional
- no acoplar el workflow a clases concretas

## Orden recomendado de implementacion en este repo

1. Refactor de contratos en `workflow`.
2. OpenAI para generacion.
3. semantic-router para `_route`.
4. Persistencia/config de rutas.
5. README y tests.

## Cambios concretos esperados por archivo

- `app/services/llm.py`
  - dividir o reemplazar
- `app/services/router.py`
  - nuevo
- `app/services/openai_llm.py`
  - nuevo
- `app/graph/workflow.py`
  - inyeccion separada de servicios
- `app/main.py`
  - nuevo cableado de dependencias
- `app/settings.py`
  - nuevas variables OpenAI y semantic-router
- `tests/test_workflow.py`
  - nuevos fakes para router y generation service
- `README.md`
  - eliminar setup de vLLM
  - documentar OpenAI + semantic-router

## Criterio de exito

La migracion se considera correcta cuando:

- no exista dependencia funcional de `vLLM`
- el router de LangGraph no use prompts de clasificacion
- las respuestas y extracciones usen OpenAI
- el router tenga rutas y thresholds versionados
- los tests cubran `conversation`, `rag`, `appointment_intent` y fallback
