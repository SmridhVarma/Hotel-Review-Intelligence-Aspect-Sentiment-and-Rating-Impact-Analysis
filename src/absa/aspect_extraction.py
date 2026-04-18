# aspect_extraction.py — Stage 2: Aspect Extraction via LDA
#
# Purpose:
#   Applies LDA topic modeling to the sentence corpus to build a keyword
#   dictionary for 6 target hotel service aspects.
#
# Input:
#   outputs/sentences.csv
#     sentence (str): sentence text to model
#
# Output:
#   outputs/aspect_dictionary.json
#     Schema: { aspect_name: [keyword, ...] }
#     Keys:   cleanliness | staff | location | noise | food | room
#     Example: { "cleanliness": ["clean", "dirty", "mold", "smell"], ... }
