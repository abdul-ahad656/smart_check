buzzfeed-webis fake news corpus 2016
====================================
the corpus comprises the output of 9 publishers in a week close to the us elections. among the selected publishers are 6 prolific hyperpartisan ones
(three left-wing and three right-wing), and three mainstream publishers (see table 1). all publishers earned facebook’s blue checkmark, indicating authenticity and an elevated status within the network. for seven weekdays (september 19 to 23 and september 26 and 27), every post and linked news article of the 9 publishers was fact-checked by professional journalists at buzzfeed. in total, 1,627 articles were checked, 826 mainstream, 256 left-wing and 545 right-wing. the imbalance between categories results from differing publication frequencies.


the corpus comes with the following files:

readme.txt            this file.
web-archives/*.warc   the web archive files that contain the http messages that where sent and received during the crawl
articles/*.xml        the articles extracted from the web archive files in xml format with annotations.
schema.xsd            schema of the article files with explanations of the used xml tags. can be used with object binding libraries (like jaxb) to parse the xml.
overview.csv          giving the portal, orientation, veracity, and url for each article. the same data is also contained in the xml files.