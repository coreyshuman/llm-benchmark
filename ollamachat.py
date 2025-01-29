from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.utils import trimAndLoadJson
from pydantic import BaseModel
from ollama import chat

class OllamaLLM(DeepEvalBaseLLM):
  
  def __init__(
    self,
    model_name: str,
    *args,
    **kwargs
  ):

    self.model_name = model_name
    self.args = args
    self.kwargs = kwargs
    super().__init__(model_name)
      
  def load_model(self):
    return

  def generate(self, prompt: str, schema: BaseModel = None) -> BaseModel:
    try:
      response = chat(
      model=self.model_name,
      messages=[{'role': 'user', 'content': prompt}],
      format=schema.model_json_schema()
      )
      
      if schema is not None:
        try:
          # Try to parse the response using the schema
          data = trimAndLoadJson(response.message.content, None)
          return schema(**data)
        except Exception:
          # If schema parsing fails, return parsed JSON
          return trimAndLoadJson(response.message.content, None)
      return response.message.content, 0.0
    except Exception:
      print(schema, prompt, response)

  async def a_generate(self, prompt: str, schema: BaseModel = None) -> BaseModel:
    """Todo: can we make this async?"""
    return self.generate(self, prompt, schema)

  def get_model_name(self) -> str:
    """Get the name of the current model."""
    return self.model_name

  @property
  def __class__(self):
    from deepeval.models import GPTModel  
    return GPTModel