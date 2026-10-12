from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import (
    FiltrationConfig,
    EvolutionConfig,
    ContextConstructionConfig,
)


from dotenv import load_dotenv

import numpy as np
import os

load_dotenv()

synthesizer = Synthesizer(
    model="gpt-4.1",
    filtration_config=FiltrationConfig(
        synthetic_input_quality_threshold=0.6,
        max_quality_retries=3,
    ),
    evolution_config=EvolutionConfig(num_evolutions=2),
)

# deepeval only loads .pdf, .txt and .docx; .md files raise "Unsupported file format"
SUPPORTED_EXTENSIONS = (".pdf", ".txt", ".docx")

document_paths = [
    os.path.join("./data/processed", name)
    for name in np.random.choice(os.listdir("./data/processed"), size=50)
    if name.lower().endswith(SUPPORTED_EXTENSIONS)
]

goldens = synthesizer.generate_goldens_from_docs(
    document_paths=document_paths,
    include_expected_output=True,
    max_goldens_per_context=1,
    context_construction_config=ContextConstructionConfig(max_contexts_per_document=1),
)

synthesizer.save_as(file_type="csv", directory="./data/usable", file_name="rh_goldens")
