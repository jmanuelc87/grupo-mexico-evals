"""GEval rubrics for the RAG eval, one prompt per quality dimension.

Each prompt is the source of truth: :func:`parse_rubric` turns its ``- a-b: outcome`` lines into
deepeval ``Rubric`` bands (0-10 integer ranges) and the rest of the text into the GEval criteria.
The ``{context}`` placeholder is not formatted here: GEval shows the test case's retrieval context
to the judge itself, so the criteria point to that field instead.
"""

from __future__ import annotations

import re

from deepeval.metrics.g_eval import Rubric

RELEVANCIA = '''\
Evalúa la RELEVANCIA de la respuesta del asistente: el grado en que aborda directamente la \
consulta del usuario.

Escala 0.0-1.0:
- 0.90-1.00: La respuesta aborda completamente la consulta con información altamente pertinente
- 0.70-0.89: La respuesta es mayormente relevante pero contiene información tangencial
- 0.50-0.69: La respuesta contiene elementos relevantes mezclados con información menos pertinente
- 0.30-0.49: La respuesta tiene cierta relevancia pero deja sin responder aspectos clave
- 0.00-0.29: La respuesta es poco relevante o no aborda la consulta

Contexto recuperado (puede estar vacío; tómalo en cuenta solo si aporta información \
relevante para la evaluación):
{context}

Devuelve la puntuación y una justificación breve en español.
'''

PRECISION = '''\
Evalúa la PRECISIÓN de la respuesta del asistente: la exactitud y corrección de los hechos, \
datos e información proporcionada.

Escala 0.0-1.0:
- 0.90-1.00: Información completamente precisa y verificable, sin errores factuales
- 0.70-0.89: Información mayormente precisa con errores menores o información ligeramente \
desactualizada
- 0.50-0.69: Mezcla de información precisa e imprecisa; algunos errores factuales presentes
- 0.30-0.49: Múltiples errores factuales; información principalmente imprecisa
- 0.00-0.29: Información fundamentalmente incorrecta o no verificable

Contexto recuperado (puede estar vacío; tómalo en cuenta solo si aporta información \
relevante para la evaluación):
{context}

Devuelve la puntuación y una justificación breve en español.
'''

COMPLETITUD = '''\
Evalúa la COMPLETITUD de la respuesta del asistente: el grado en que cubre todos los aspectos \
relevantes de la pregunta.

Escala 0.0-1.0:
- 0.90-1.00: Respuesta exhaustiva que cubre todos los aspectos principales y secundarios relevantes
- 0.70-0.89: Cubre los aspectos principales; faltan algunos detalles secundarios
- 0.50-0.69: Cubre aproximadamente la mitad de los aspectos relevantes
- 0.30-0.49: Cubre solo algunos aspectos; hay grandes vacíos de información
- 0.00-0.29: Respuesta incompleta que cubre muy poco del tema

Contexto recuperado (puede estar vacío; tómalo en cuenta solo si aporta información \
relevante para la evaluación):
{context}

Devuelve la puntuación y una justificación breve en español.
'''

CLARIDAD = '''\
Evalúa la CLARIDAD de la respuesta del asistente: la facilidad de comprensión y la calidad de \
la comunicación y del texto.

Escala 0.0-1.0:
- 0.90-1.00: Extremadamente clara, bien estructurada, con lenguaje preciso y fácil de seguir
- 0.70-0.89: Mayormente clara, estructura coherente, con lenguaje generalmente preciso
- 0.50-0.69: Moderadamente clara; algunas partes son confusas o desorganizadas
- 0.30-0.49: Frecuentemente confusa; estructura pobre; lenguaje impreciso
- 0.00-0.29: Muy confusa, desorganizada, casi imposible de entender

Contexto recuperado (puede estar vacío; tómalo en cuenta solo si aporta información \
relevante para la evaluación):
{context}

Devuelve la puntuación y una justificación breve en español.
'''

RAZONAMIENTO_LOGICO = '''\
Evalúa el RAZONAMIENTO LÓGICO de la respuesta del asistente: la calidad de la lógica, la \
coherencia de los argumentos y la justificación de las conclusiones.

Escala 0.0-1.0:
- 0.90-1.00: Razonamiento sólido, argumentos bien justificados, conclusiones lógicamente válidas
- 0.70-0.89: Razonamiento mayormente sólido con justificaciones adecuadas
- 0.50-0.69: Razonamiento aceptable pero con algunas inconsistencias lógicas
- 0.30-0.49: Razonamiento débil con varias inconsistencias o saltos lógicos
- 0.00-0.29: Razonamiento falso o lógicamente incoherente

Contexto recuperado (puede estar vacío; tómalo en cuenta solo si aporta información \
relevante para la evaluación):
{context}

Devuelve la puntuación y una justificación breve en español.
'''

CONTEXTUALIZACION = '''\
Evalúa la CONTEXTUALIZACIÓN de la respuesta del asistente: la comprensión y el uso efectivo del \
contexto proporcionado en la consulta y el escenario.

Escala 0.0-1.0:
- 0.90-1.00: Demuestra comprensión profunda del contexto; lo integra de manera experta
- 0.70-0.89: Entiende y usa bien el contexto en la mayoría de la respuesta
- 0.50-0.69: Reconoce el contexto pero no lo integra completamente
- 0.30-0.49: Comprensión limitada del contexto; referencias inadecuadas
- 0.00-0.29: Ignora o malentiende el contexto completamente

Contexto recuperado (puede estar vacío; tómalo en cuenta solo si aporta información \
relevante para la evaluación):
{context}

Devuelve la puntuación y una justificación breve en español.
'''

ACCIONABILIDAD = '''\
Evalúa la ACCIONABILIDAD de la respuesta del asistente: la utilidad práctica de la información \
y qué tan aplicable y ejecutable resulta.

Escala 0.0-1.0:
- 0.90-1.00: Información altamente práctica y ejecutable; pasos claros para implementación
- 0.70-0.89: Información mayormente práctica con aplicación clara
- 0.50-0.69: Contiene información útil pero con aplicación limitada
- 0.30-0.49: Poca aplicabilidad práctica; información mayormente teórica
- 0.00-0.29: No aplicable; no proporciona dirección práctica

Contexto recuperado (puede estar vacío; tómalo en cuenta solo si aporta información \
relevante para la evaluación):
{context}

Devuelve la puntuación y una justificación breve en español.
'''

ESTRUCTURA = '''\
Evalúa la ESTRUCTURA de la respuesta del asistente: la organización, el formato y la \
presentación lógica de la información.

Escala 0.0-1.0:
- 0.90-1.00: Estructura excelente con secciones claras, uso de encabezados o listas cuando es \
apropiado
- 0.70-0.89: Estructura clara y lógica, generalmente bien organizada
- 0.50-0.69: Estructura aceptable pero con algunas desorganizaciones
- 0.30-0.49: Estructura pobre; información desorganizada o difícil de seguir
- 0.00-0.29: Estructura muy deficiente; prácticamente sin organización

Contexto recuperado (puede estar vacío; tómalo en cuenta solo si aporta información \
relevante para la evaluación):
{context}

Devuelve la puntuación y una justificación breve en español.
'''

PROFUNDIDAD_ANALITICA = '''\
Evalúa la PROFUNDIDAD ANALÍTICA de la respuesta del asistente: el nivel de análisis y la \
profundidad en la exploración del tema.

Escala 0.0-1.0:
- 0.90-1.00: Análisis profundo y perspicaz; explora múltiples dimensiones del problema
- 0.70-0.89: Análisis sólido con exploración adecuada de aspectos clave
- 0.50-0.69: Análisis moderado; algunos aspectos explorados superficialmente
- 0.30-0.49: Análisis superficial; mayormente respuestas de nivel básico
- 0.00-0.29: Análisis mínimo o inexistente; respuesta superficial

Contexto recuperado (puede estar vacío; tómalo en cuenta solo si aporta información \
relevante para la evaluación):
{context}

Devuelve la puntuación y una justificación breve en español.
'''

COHERENCIA_MULTITURNO = '''\
Evalúa la COHERENCIA MULTI-TURNO de la respuesta del asistente: su consistencia y coherencia a \
lo largo de los turnos previos de la conversación, que se incluyen más abajo.

Escala 0.0-1.0:
- 0.90-1.00: Perfectamente coherente a través de todos los turnos; mantiene consistencia temática
- 0.70-0.89: Generalmente coherente; mantiene la mayoría de elementos consistentes
- 0.50-0.69: Coherencia aceptable con algunas inconsistencias menores
- 0.30-0.49: Varias inconsistencias; el hilo se pierde en algunos puntos
- 0.00-0.29: Incoherente; numerosas contradicciones entre turnos

Contexto recuperado (puede estar vacío; tómalo en cuenta solo si aporta información \
relevante para la evaluación):
{context}

Devuelve la puntuación y una justificación breve en español.
'''

# Single-turn dimensions, by GEval metric name. COHERENCIA_MULTITURNO is kept above for
# reference but left out: the RAG eval asks one question per test case, so there are no turns.
RUBRICS = {
    "relevancia": RELEVANCIA,
    "precision": PRECISION,
    "completitud": COMPLETITUD,
    "claridad": CLARIDAD,
    "razonamiento_logico": RAZONAMIENTO_LOGICO,
    "contextualizacion": CONTEXTUALIZACION,
    "accionabilidad": ACCIONABILIDAD,
    "estructura": ESTRUCTURA,
    "profundidad_analitica": PROFUNDIDAD_ANALITICA,
}

_BAND = re.compile(r"^- (\d\.\d\d)-(\d\.\d\d): (.+)$")
_CONTEXT_PLACEHOLDER = "{context}"
_CONTEXT_POINTER = "se proporciona en el campo Retrieval Context."


def parse_rubric(prompt: str) -> tuple[str, list[Rubric]]:
    """Split a prompt into (criteria, rubric bands).

    Bands ``0.90-1.00`` become deepeval's integer range ``(9, 10)``, ``0.70-0.89`` becomes
    ``(7, 8)``, and so on; GEval divides the 0-10 score by 10, so thresholds stay on 0-1.
    """
    criteria_lines: list[str] = []
    rubric: list[Rubric] = []
    for line in prompt.splitlines():
        band = _BAND.match(line)
        if band:
            low, high, outcome = band.groups()
            rubric.append(
                Rubric(
                    score_range=(round(float(low) * 10), int(float(high) * 10)),
                    expected_outcome=outcome,
                )
            )
        elif line.startswith("Escala "):
            continue  # the bands carry the scale
        else:
            criteria_lines.append(line.replace(_CONTEXT_PLACEHOLDER, _CONTEXT_POINTER))
    criteria = re.sub(r"\n{3,}", "\n\n", "\n".join(criteria_lines)).strip()
    return criteria, rubric
