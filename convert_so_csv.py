import lxml
import os
import pandas as pd
import csv
from xml.etree import cElementTree
from lxml import etree

POSTS_XML_PATH = "../data/stackoverflow/Posts.xml"
POSTS_CSV_PATH = "../formatted_data/stackoverflow/posts.csv"
LINK_XML_PATH = "../data/stackoverflow/PostLinks.xml"
LINK_CSV_PATH = "../formatted_data/stackoverflow/links.csv"

def iter_elements_by_name(handle, name):
    events = cElementTree.iterparse(handle, events=("start", "end",))
    _, root = next(events)  # Grab the root element.
    for event, elem in events:
        if event == "end" and elem.tag == name:
            yield elem
            root.clear()  # Free up memory by clearing the root element.

            
def iter_rows(handle):
    for row in iter_elements_by_name(handle, "row"):
        yield row.attrib
        
        
        
if __name__ == "__main__":

  post_iterator = iter_rows(POSTS_XML_PATH)

  for row in post_iterator:
      print(row)
      break

  columns = list(row.keys())
  print(columns)


  post_iterator = iter_rows(POSTS_XML_PATH)

  with open(POSTS_CSV_PATH, "w") as outf:
      writer = csv.writer(outf, delimiter="|")
      writer.writerow(columns)
      for i, row in enumerate(post_iterator):
          if i % 100000 == 0:
              print(i)
          writer.writerow([row.get(c) for c in columns])

  link_tree = etree.parse(LINK_XML_PATH)
  link_df = pd.DataFrame(data=[dict(child.attrib.items()) for child in link_tree.iter("row")])
  link_df.to_csv(LINK_CSV_PATH, sep="|", index=False)
