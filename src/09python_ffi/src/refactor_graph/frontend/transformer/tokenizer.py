import numpy as np
from regex import F


class InfiniTensorTokenizer:
    def encode(self, input_text: str) -> np.ndarray:
        raise NotImplementedError()

    def decode(self, token_ids: np.ndarray) -> str:
        raise NotImplementedError()


class HuggingFaceTokenizer(InfiniTensorTokenizer):
    def __init__(self, model_path) -> None:
        from transformers import AutoTokenizer

        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True
        )

    def encode(self, input_text):
        return self.tokenizer.encode(input_text, return_tensors="np")

    def decode(self, token_ids):
        # TODO: AutoTokenizer cannot covert "▁" to white space.
        #       Should use decode method when fixed.
        if isinstance(token_ids, int):
            return self.tokenizer.convert_ids_to_tokens(token_ids)
        else:
            result = ""
            for s in self.tokenizer.convert_ids_to_tokens(token_ids):
                result += s.replace("▁", " ")
            return result
