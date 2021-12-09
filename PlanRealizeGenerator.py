import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class PlanRealizeGenerator:
    def __init__(self, planner_path, realizer_path):
        self.planner_tokenizer = T5Tokenizer.from_pretrained(planner_path)
        self.planner = T5ForConditionalGeneration.from_pretrained(planner_path).to(DEVICE)
        self.realizer_tokenizer = T5Tokenizer.from_pretrained(realizer_path)
        self.realizer = T5ForConditionalGeneration.from_pretrained(realizer_path).to(DEVICE)
    
    def generate(self, data_entries):
        planner_input_ids = self.planner_tokenizer(data_entries['input'])
        pass



if __name__ == '__main__':
    model = PlanRealizeGenerator(planner_path='royeis/T5-FlowNLG-Planner', realizer_path='royeis/T5-FlowNLG-Realizer')
    print('done')