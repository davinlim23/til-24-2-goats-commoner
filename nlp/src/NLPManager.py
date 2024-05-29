import torch
from typing import Dict
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
# import sys

class NLPManager:
    def __init__(self):
        # load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("best_model")
        self.model = AutoModelForTokenClassification.from_pretrained("best_model")
        self.pipe = pipeline("ner", model=self.model, tokenizer=self.tokenizer, device_map="auto", aggregation_strategy="max")
        
        # convert heading to numeric
        self.number_to_word = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "niner"]
        self.word_to_number = {}
        for idx, word in enumerate(self.number_to_word):
            self.word_to_number[word] = idx

    def qa(self, context: str) -> Dict[str, str]:
        # perform NLP question-answering
        out = self.pipe(context)
        output = self.postprocess(out)
        
        return output
    
    def postprocess(self, predictions):
        output = {"heading": "", "tool": "", "target": ""}
        
        # format predictions to output format
        for prediction in predictions:
            if prediction["entity_group"] == "TOOL" and output["tool"] == "":
                output["tool"] = prediction["word"]
            elif prediction["entity_group"] == "TAR" and output["target"] == "":
                output["target"] = prediction["word"]
            elif prediction["entity_group"] == "HEAD" and output["heading"] == "":
                heading_words = prediction["word"].split(" ")
                
                for word in heading_words:
                    try:
                        output["heading"] += str(self.word_to_number[word])
                    except:
                        pass
            else:
                pass
        
        return output        
    
# if __name__ == "__main__":
#     manager = NLPManager()
#     print(manager.qa(sys.argv[1]))
