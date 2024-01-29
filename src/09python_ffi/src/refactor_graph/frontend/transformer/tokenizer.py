import numpy as np
from regex import F
from typing import List


class InfiniTensorTokenizer:
    def encode(self, input_text: str) -> List[int]:
        raise NotImplementedError()

    def decode(self, token_ids: int | List[int]) -> str:
        raise NotImplementedError()


class HuggingFaceTokenizer(InfiniTensorTokenizer):
    def __init__(self, model_path) -> None:
        from transformers import AutoTokenizer

        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True
        )

    def encode(self, input_text) -> List[int]:
        return self.tokenizer.encode(input_text)

    def decode(self, token_ids) -> str:
        if isinstance(token_ids, int):
            return self.tokenizer.convert_ids_to_tokens(token_ids).replace("▁", " ")
        elif isinstance(token_ids, np.ndarray):
            token_ids = token_ids.flatten().tolist()
        result = ""
        for s in self.tokenizer.convert_ids_to_tokens(token_ids):
            # TODO: AutoTokenizer cannot covert "▁" to white space.
            #       Should use decode method when fixed.
            result += s.replace("▁", " ")
        return result
