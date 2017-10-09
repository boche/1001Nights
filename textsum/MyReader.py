import gzip
import sys
import re
import glob
import random
import xml.etree.ElementTree as ET

class MyReader:
    def __init__(self, pattern):
        self.fid = 0
        self.filenames = glob.glob(pattern)
        random.shuffle(self.filenames)
        self.f = gzip.open(self.filenames[self.fid], 'r')
        
    def textify(self, content):
        return [x.lower() for x in re.findall("([^\s)]+)\)", content)]
    
    def parse_doc(self, content):
        root = ET.fromstring(content)
        docid = root.attrib['id']
        head = []
        headline = root.find("HEADLINE")
        if headline is not None:
            head = self.textify(headline.text)
        body = []
        text = root.find("TEXT")
        if text is not None:
            for para in text:
                body.extend(self.textify(para.text))
        return (docid, head, body)
    
    def gen_docs(self):
        content = ""
        while True:
            line = self.f.readline().decode("utf-8")
            if len(line) == 0:
                # indicate end of file
                self.f.close()
                self.fid += 1
                if self.fid == len(self.filenames):
                    break
                self.f = gzip.open(self.filenames[self.fid], 'r')
                continue
            if "<DOC " in line:
                content = ""
            content += line
            if "</DOC>" in line:
                yield self.parse_doc(content)
