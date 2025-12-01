from lxml import etree

XML_FILE = r"xml/Votes.xml" 

count = 0
for _, elem in etree.iterparse(XML_FILE, events=("end",), tag="row"):
    if elem.attrib["VoteTypeId"] == "3":
        count += 1
    elem.clear()
    while elem.getprevious() is not None:
        del elem.getparent()[0]

print("Total <row> count:", count)