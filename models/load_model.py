# load_model_and_tokenizer.py

from models.mistral_model.model import mistral_load  
from models.madlad_model.model import madlad_load
from models.m2m_model.model import m2m_load
from models.nllb_model.model import nllb_load
from models.tower_instruct_model.model import towerinstruct_load
from models.helsinki_model.model import helsinki_load

def load_model_and_tokenizer(model_name="mistral"):
    match model_name:
        case "mistral":
            return mistral_load()
        case "madlad":
            return madlad_load()
        case "m2m":
            return m2m_load()
        case "nllb":
            return nllb_load()
        case "tower instruct":
            return towerinstruct_load()
        case "helsinki":
            return helsinki_load()
        case _:
            raise ValueError(f"Model {model_name} is not supported in this hub.")
