def triple_to_string(triple):
    return '<S> ' + triple.subject + ' <P> ' + triple.predicate + ' <O> ' + triple.object

def tripleset_to_string(tripleset):
    triple_strings = [triple_to_string(t) for t in tripleset]
    return ' '.join(triple_strings)

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



def deepnlg_entry_to_examples(entry):
    examples = []
    for lex in entry.lexEntries:
        example = {}
        example['category'] = entry.category
        example['eid'] = entry.eid
        example['size'] = entry.size
        example['planner_input'] = tripleset_to_string(entry.modifiedtripleset)
        example['lid'] = lex.lid
        example['text'] = lex.text
        example['plan'] = get_plan(entry.modifiedtripleset, lex.orderedtripleset)
        example['realizer_input'] = ordered_tripleset_to_string(lex.orderedtripleset)
        examples.append(example)
    return examples