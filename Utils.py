import torch
from sacrebleu import corpus_bleu

def triple_to_string(triple):
    return '<S> ' + triple.subject + ' <P> ' + triple.predicate + ' <O> ' + triple.object

def tripleset_to_string(tripleset):
    triple_strings = [triple_to_string(t) for t in tripleset]
    return ' '.join(triple_strings)

def tripleset_map(tripleset):
    triple_strings = [triple_to_string(t) for t in tripleset]
    return {i: triple for i, triple in enumerate(triple_strings)}

def get_plan(original_tripleset, ordered_tripleset):
    triple_strings = [triple_to_string(t) for t in original_tripleset]
    ordered_triple_strings = [[triple_to_string(t) for t in sent] for sent in ordered_tripleset]
    
    sentence_plan_strings = []
    for ordered_sentence in ordered_triple_strings:
        sentence_string = 'S'
        for ordered_triple in ordered_sentence:
            for i, triple in enumerate(triple_strings):
                if ordered_triple == triple:
                    sentence_string += ' ' + str(i)
                    break
        
        sentence_plan_strings.append(sentence_string)
    
    return ' '.join(sentence_plan_strings)

def ordered_tripleset_to_string(ordered_tripleset):
    sentence_triplesets = ['<sentence> ' + tripleset_to_string(t_set) for t_set in ordered_tripleset]
    return ' '.join(sentence_triplesets)

def webnlg_entry_to_examples(entry):
    examples = []
    for lex in entry.lexs:
        example = {}
        example['category'] = entry.category
        example['eid'] = entry.id
        example['size'] = entry.size
        example['triples_map'] = tripleset_map(entry.modifiedtripleset.triples)
        example['input'] = tripleset_to_string(entry.modifiedtripleset.triples)
        example['lid'] = lex.id
        example['text'] = lex.lex
        examples.append(example)

    return examples

def deepnlg_entry_to_examples(entry):
    examples = []
    for lex in entry.lexEntries:
        example = {}
        example['category'] = entry.category
        example['eid'] = entry.eid
        example['size'] = entry.size
        example['triples_map'] = tripleset_map(entry.modifiedtripleset)
        example['planner_input'] = tripleset_to_string(entry.modifiedtripleset)
        example['lid'] = lex.lid
        example['text'] = lex.text
        example['plan'] = get_plan(entry.modifiedtripleset, lex.orderedtripleset)
        example['realizer_input'] = ordered_tripleset_to_string(lex.orderedtripleset)
        examples.append(example)

    return examples


def validate_plan(plan, n_triples):
    plan_set = set(plan.split())
    plan_set.remove('S')
    expected = set([str(i) for i in range(n_triples)])
    return plan_set == expected

class PlannerDataset(torch.utils.data.Dataset):
    def __init__(self, examples, tokenizer):
        self.tokenized_input = tokenizer([ex['planner_input'] for ex in examples], padding=True, return_tensors='pt')
        self.labels = tokenizer([ex['plan'] for ex in examples], padding=True, return_tensors='pt')

    def __getitem__(self, idx):
        return {
                'input_ids': self.tokenized_input['input_ids'][idx],
                'attention_mask': self.tokenized_input['attention_mask'][idx],
                'labels': self.labels['input_ids'][idx],
                'decoder_attention_mask': self.labels['attention_mask'][idx]
                }

    def __len__(self):
        return len(self.tokenized_input['input_ids'])


class RealizerDataset(torch.utils.data.Dataset):
    def __init__(self, examples, tokenizer):
        self.tokenized_input = tokenizer([ex['realizer_input'] for ex in examples], padding=True, return_tensors='pt')
        self.labels = tokenizer([ex['text'] for ex in examples], padding=True, return_tensors='pt')

    def __getitem__(self, idx):
        return {
                'input_ids': self.tokenized_input['input_ids'][idx],
                'attention_mask': self.tokenized_input['attention_mask'][idx],
                'labels': self.labels['input_ids'][idx],
                'decoder_attention_mask': self.labels['attention_mask'][idx]
                }

    def __len__(self):
        return len(self.tokenized_input['input_ids'])



def T2TDataCollator(batch):
    """
    Take a list of samples from a Dataset and collate them into a batch.
    Returns:
        A dictionary of tensors
    """
    input_ids = torch.stack([example['input_ids'] for example in batch])
    lm_labels = torch.stack([example['labels'] for example in batch])
    lm_labels[lm_labels[:, :] == 0] = -100
    attention_mask = torch.stack([example['attention_mask'] for example in batch])
    decoder_attention_mask = torch.stack([example['decoder_attention_mask'] for example in batch])

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': lm_labels,
        'decoder_attention_mask': decoder_attention_mask
    }

def compute_bleu(generations, references):
    max_entry_refs = max([len(r) for r in references])
    grouped_refs = [[] for i in range(max_entry_refs)]
    for ref_group in references:
        for i in range(max_entry_refs):
            if i < len(ref_group):
                grouped_refs[i].append(ref_group[i])
            else:
                grouped_refs[i].append('')
    
    return corpus_bleu(generations, grouped_refs).score