import gzip
import sys
import re
import xml.etree.ElementTree as ET

def textify(content):
  words = re.findall("([^\s)]+)\)", content)
  return " ".join(words)

def parse_doc(content):
  root = ET.fromstring(content)
  headline = root.find("HEADLINE")
  text = root.find("TEXT")
  if headline is not None:
    print(textify(headline.text))
  if text is not None:
    for para in text:
      print(textify(para.text))
  print("------")

def load_file(filename):
  content = ""
  with gzip.open(filename, 'r') as f:
    for line in f:
      line = line.decode("utf-8")
      if "<DOC " in line:
        content = ""
      content += line
      if "</DOC>" in line:
        parse_doc(content)

if __name__ == "__main__":
  load_file(sys.argv[1])
