from WebNLGDatasetReader import Benchmark, select_files

# where to find the corpus
path_to_corpus = 'webnlg_v3/en/test'

# initialise Benchmark object
b = Benchmark()

# collect xml files
if path_to_corpus.split('/')[-1] == 'test':
    files = [(path_to_corpus, 'rdf-to-text-generation-test-data-with-refs-en.xml')]
else:
    files = select_files(path_to_corpus)

# load files to Benchmark
b.fill_benchmark(files)

# output some statistics
print("Number of entries: ", b.entry_count())
print("Number of texts: ", b.total_lexcount())
print("Number of distinct properties: ", len(list(b.unique_p_mtriples())))

# get access to each entry info
for entry in b.entries:
    print(f"Info about {entry.id} in category '{entry.category}' in size '{entry.size}':")
    print("# of lexicalisations", entry.count_lexs())
    print("Properties: ", entry.relations())
    print("RDF triples: ", entry.list_triples())
    print("Subject:", entry.modifiedtripleset.triples[0].s)
    print("Predicate:", entry.modifiedtripleset.triples[0].p)
    print("Object:", entry.modifiedtripleset.triples[0].o)
    print("Lexicalisation:", entry.lexs[0].lex)
    print("Another lexicalisation:", entry.lexs[1].lex)
    if entry.dbpedialinks:
        # dbpedialinks is a list where each element is a Triple instance
        print("DB link, en:", entry.dbpedialinks[0].s)  # subject in English
        print("DB link, ru:", entry.dbpedialinks[0].o)  # object in Russian
    break
