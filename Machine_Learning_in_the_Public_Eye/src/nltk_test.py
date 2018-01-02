import nltk


print("\n\n\nList of imported Texts\n")
from nltk.book import *

print("\n\n\nExample use of concordance in text 1\n")
text1.concordance("monstrous")

print("\n\n\nExample use of concordance in text 2\n")
text2.concordance("monstrous")

print("\n\n\nExample use of similar in text 2\n")
text2.similar("monstrous")


print("\n\n\nExample use of common_contexts in text 2\n")
text2.common_contexts(["monstrous", "very"])


text4.dispersion_plot(["citizens","democracy","freedom","duties","America"])
