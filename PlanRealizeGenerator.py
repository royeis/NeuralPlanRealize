import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class PlanRealizeGenerator:
    def __init__(self, planner_path, realizer_path):
        self.planner_tokenizer = T5Tokenizer.from_pretrained(planner_path)
        self.planner = T5ForConditionalGeneration.from_pretrained(planner_path)
        self.realizer_tokenizer = T5Tokenizer.from_pretrained(realizer_path)
        self.realizer = T5ForConditionalGeneration.from_pretrained(realizer_path)
    
    def _plan(self, input_data_string):
        """
        Generates a string of plan encoding for a given data input string
        :param input_data_string: a string representing the triples in the input data
        :return: plan encoding
        """
        planner_input_ids = self.planner_tokenizer(input_data_string, return_tensors='pt')
        planner_input_ids.to(DEVICE)
        plan_encoding_ids = self.planner.generate(**planner_input_ids)

        return self.planner_tokenizer.decode(plan_encoding_ids[0], skip_special_tokens=True)

    def _realize(self, plan_string):
        """
        Generates natural language description of the data given a plan
        :param plan_string: a string representing the plan for generation
        :returns: natural language description of the input data
        """
        
        realizer_input_ids = self.realizer_tokenizer(plan_string, return_tensors='pt')
        realizer_input_ids.to(DEVICE)
        output_ids = self.realizer.generate(**realizer_input_ids, max_length=256)

        return self.realizer_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    def generate_plan(self, input_example):
        plan_encoding = self._plan(input_example['planner_input'])
        plan_encoding = self._validate_plan(plan_encoding, int(input_example['size']))
        return self._plan_encoding_to_string(input_example['triples_map'], plan_encoding)
    
    def generate_realization(self, plan_string):
        return self._realize(plan_string)

    def plan_mode(self):
        self.realizer.cpu()
        self.planner.to(DEVICE)
    
    def realize_mode(self):
        self.planner.cpu()
        self.realizer.to(DEVICE)

    def generate(self, input_example):
        """
        Generates natural language description of the data given data input
        :param input_data_string: a string representing the triples in the input data
        :return: natural language description of the input data
        """
        self.plan_mode()
        plan_string = self.generate_plan(input_example)

        self.realize_mode()
        output = self.generate_realization(plan_string)
        return output

    def _validate_plan(self, plan, n_triples):
        plan_set = set(plan.split())
        plan_set.remove('S')
        expected = set([str(i) for i in range(n_triples)])
        
        extra_syms = plan_set - expected
        if extra_syms:
            for sym in extra_syms:
                plan = plan.replace(sym, '')
            
            plan = plan.replace('  ', ' ')
            plan = plan.rstrip()
        
        missing_syms = expected - plan_set
        if missing_syms:
            missing_sent = f"S {' '.join(missing_syms)}"
            plan = f"{plan} {missing_sent}"
        
        plan = plan.rstrip('S ')
        
        return plan


    def _plan_encoding_to_string(self, triples_map, plan_encoding):
        plan_string = ''
        for c in plan_encoding:
            if c == ' ':
                plan_string += ' '
            elif c == 'S':
                plan_string += '<sentence>'
            elif int(c) in triples_map.keys():
                plan_string += triples_map[int(c)]
            else:
                raise KeyError(f'unexpected key: {c}')
        return plan_string


if __name__ == '__main__':
    model = PlanRealizeGenerator(planner_path='royeis/T5-FlowNLG-Planner', realizer_path='royeis/T5-FlowNLG-Realizer')
    print('done')