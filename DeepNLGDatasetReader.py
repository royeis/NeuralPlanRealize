import os
import xml.etree.ElementTree as ET


class Entry():
    def __init__(self, category, eid, size, originaltripleset, modifiedtripleset, entitymap, lexEntries):
        self.category = category
        self.eid = eid
        self.size = size
        self.originaltripleset = originaltripleset
        self.modifiedtripleset = modifiedtripleset
        self.lexEntries = lexEntries
        self.entitymap = entitymap

    def entitymap_to_dict(self):
        return dict(map(lambda tagentity: tagentity.to_tuple(), self.entitymap))

class Triple():
    def __init__(self, subject, predicate, object):
        self.subject = subject
        self.predicate = predicate
        self.object = object

class Lex():
    def __init__(self, comment, lid, text, template, orderedtripleset=[], references=[]):
        self.comment = comment
        self.lid = lid
        self.text = text
        self.template = template
        self.tree = ''
        self.orderedtripleset = orderedtripleset
        self.references = references

        # german entry
        self.text_de = ''
        self.template_de = ''
        self.tree_de = ''
        self.orderedtripleset_de = []
        self.references_de = []

class TagEntity():
    def __init__(self, tag, entity):
        self.tag = tag
        self.entity = entity

    def to_tuple(self):
        return (self.tag, self.entity)

class Reference():
    def __init__(self, tag, entity, refex, number, reftype):
        self.tag = tag
        self.entity = entity
        self.refex = refex
        self.number = number
        self.reftype = reftype

def parse(in_file):
    tree = ET.parse(in_file)
    root = tree.getroot()

    entries = root.find('entries')

    for entry in entries:
        eid = entry.attrib['eid']
        size = entry.attrib['size']
        category = entry.attrib['category']

        originaltripleset = []
        otripleset = entry.find('originaltripleset')
        for otriple in otripleset:
            e1, pred, e2 = otriple.text.split(' | ')
            originaltripleset.append(Triple(subject=e1.replace('\'', ''), predicate=pred, object=e2.replace('\'', '')))

        modifiedtripleset = []
        mtripleset = entry.find('modifiedtripleset')
        for mtriple in mtripleset:
            e1, pred, e2 = mtriple.text.split(' | ')

            modifiedtripleset.append(Triple(subject=e1.replace('\'', ''), predicate=pred, object=e2.replace('\'', '')))

        entitymap = []
        mapping= entry.find('entitymap')
        for entitytag in mapping:
            tag, entity = entitytag.text.split(' | ')
            entitymap.append(TagEntity(tag=tag, entity=entity))

        lexList = []
        lexEntries = entry.findall('lex')
        for lex in lexEntries:
            comment = lex.attrib['comment']
            lid = lex.attrib['lid']

            try:
                orderedtripleset = []
                otripleset = lex.find('sortedtripleset')
                for snt in otripleset:
                    orderedtripleset_snt = []
                    for otriple in snt:
                        e1, pred, e2 = otriple.text.split(' | ')

                        orderedtripleset_snt.append(Triple(subject=e1.replace('\'', ''), predicate=pred, object=e2.replace('\'', '')))
                    orderedtripleset.append(orderedtripleset_snt)
            except:
                orderedtripleset = []

            try:
                references = []
                references_xml = lex.find('references')
                for ref in references_xml:
                    tag = ref.attrib['tag']
                    entity = ref.attrib['entity']
                    number = ref.attrib['number']
                    reftype = ref.attrib['type']
                    refex = ref.text
                    references.append(Reference(tag=tag, entity=entity, number=number, reftype=reftype, refex=refex))
            except:
                references = []

            try:
                text = lex.find('text').text
                if not text:
                    text = ''
            except:
                print('exception text')
                text = ''

            try:
                template = lex.find('template').text
                if not template:
                    template = ''
            except:
                print('exception template')
                template = ''

            lexList.append(Lex(comment=comment, lid=lid, text=text, template=template, orderedtripleset=orderedtripleset, references=references))

        yield Entry(eid=eid, size=size, category=category, originaltripleset=originaltripleset, \
                    modifiedtripleset=modifiedtripleset, entitymap=entitymap, lexEntries=lexList)

def run_parser(set_path):
    entryset = []
    dirtriples = filter(lambda item: not str(item).startswith('.'), os.listdir(set_path))
    for dirtriple in dirtriples:
        fcategories = filter(lambda item: not str(item).startswith('.'), os.listdir(os.path.join(set_path, dirtriple)))
        for fcategory in fcategories:
            entryset.extend(list(parse(os.path.join(set_path, dirtriple, fcategory))))

    return entryset
