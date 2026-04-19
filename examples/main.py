from src.downloads import ensure_model
from src.cactus import cactus_init, cactus_complete, cactus_destroy
import json

# From the cactus repo (after `source ./setup`): `cactus download google/gemma-3-270m-it --reconvert`
# If weights are missing locally, ensure_model() may fetch pre-converted INT4 zips from HuggingFace.
MODEL_ID = "google/gemma-3-270m-it"
weights = ensure_model(MODEL_ID)

model = cactus_init(str(weights), None, False)
messages = json.dumps(
    [{"role": "user", "content": "How many seconds are in a minute?"}]
)
result = json.loads(cactus_complete(model, messages, None, None, None))
print(result["response"])
cactus_destroy(model)
