# TextTransfer
Prior analyses and assessments of the impact of scientific research has mainly relied on analyzing its scope within academia and its influence within scholarly circles. Nevertheless, by not considering the broader societal, economic, and policy implications of research projects, these studies overlook the ways in which scientific discoveries contribute to technological innovation, public health improvements, environmental sustainability, and other areas of real-world application. We expand upon this prior work by developing and validating a conceptual and computational solution to automatically identify and categorize scientific research's impact within and especially beyond academia based on text data. 

We first propose a framework for automatically assessing the impact of scientific research projects by identifying pertinent sections in project reports that indicate the potential impacts. We leverage a mixed-method approach, combining manual annotations with supervised machine learning, to extract these passages from project reports. We then empirically develop and evaluate an annotation schema to capture and classify the impact of research projects based on research reports from different scientific domains. We then annotate a large dataset of over 45k sentences extracted from research reports for the developed impact categories. We examine the annotated dataset for patterns in the distribution of impact categories across different scientific domains, co-occurrences of impact categories, and signal words of impact. Using the annotated texts and the novel classification schema, we investigate the performance of large language models (LLMs) for automated impact classification. Our results show that fine-tuning the models on our annotated datasets statistically significantly outperforms zero- and few-shot prompting approaches. This indicates that state-of-the-art LLMs without finetuning may not work well for novel classification schemas such as our impact classification schema, and in turn highlights the importance of diligent manual annotations as empirical basis in the field of computational social science.

This is a repository to save datasets and codes related to this project.

## Impact-relevant passage detection (passage_detection)
Please read and cite the following paper if you would like to use the data:

Becker M., Han K., Werthmann A., Rezapour R., Lee H., Diesner J., and Witt A. (2024). Detecting Impact Relevant Sections in Scientific Research. The 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING).

You can find the data at https://doi.org/10.13012/B2IDB-9934303_V1.

### Pdf to Text
Please use pdf_to_text.ipynb to convert pdf files to txt files.
After conversion, you can restructure the folder as folder "texttransfer data" -> (domain1 subfolder, domain2 subfolder, domain3 subfolder, domain4 subfolder): each domain folder contains the text files of reports (one text file per report). Only in this way can you run the second notebook (do not forget to modify the dir you store the "texttransfer data" folder).

### data clean and restructure
Data cleaning and restructuring.ipynb is a notebook for data cleaning and restructure. It will read all text files and split them into paragraphs. It will generate two dataset: ttparagraph_addmob.txt.gz is the corpus while impact_paragraph.xlsx is the pilot study data (for annotation and model training).

### paragraph extraction
paragraph_extraction.ipynb is a "long" notebook for paragraph extraction, including rule-based extraction and random forest model training and prediction, as well as datasets merging (TT-I + TT-II).

## Impact annotation (impact_annotation)
Please read the codebook IMPACT-Codebook-release.pdf.

## Impact classification (impact_classification)

### Llama and GPT Prompting
For Llama models, we used replicate's API. Please see the code at: https://replicate.com/meta/llama-2-70b-chat/api

For ChatGPT and GPT-4, we used OpenAI's API. Please see the code at: https://platform.openai.com/docs/guides/text-generation

The most import hyperparameter is temperature. We set it as 0.99 for Llama models and GPT models.

### Fine-tuning


