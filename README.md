# texttransfer
## Pdf to Text
Please use pdf_to_text.ipynb to convert pdf files to txt files.
After conversion, you can restructure the folder as folder "texttransfer data" -> (domain1 subfolder, domain2 subfolder, domain3 subfolder, domain4 subfolder): each domain folder contains the text files of reports (one text file per report). Only in this way can you run the second notebook (do not forget to modify the dir you store the "texttransfer data" folder).
## data clean and restructure
data cleaning and restructuring.ipynb is a notebook for data cleaning and restructure. It will read all text files and split them into paragraphs. It will generate two dataset: ttparagraph_addmob.txt.gz is the corpus while impact_paragraph.xlsx is the pilot study data (for annotation and model training).
