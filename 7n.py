import nltk
nltk.download('punkt_tab')   # one-time download

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers    import Tokenizer
from sumy.summarizers.lex_rank  import LexRankSummarizer   # → Default
from sumy.summarizers.luhn      import LuhnSummarizer       # → Creative
from sumy.summarizers.lsa       import LsaSummarizer        # → Structured
from sumy.summarizers.text_rank import TextRankSummarizer   # → Diverse

text = """The Industrial Revolution, which took place from the 18th to the 19th centuries, 
was a period during which predominantly agrarian, 
rural societies in Europe and America became industrial and urban.

Prior to the Industrial Revolution, manufacturing was often 
done in people's homes, using hand tools or basic machines. 
Industrialization marked a shift to powered, special-purpose machinery, factories and mass production. 

The iron and textile industries, along with the development of the steam engine, 
played central roles in the Industrial Revolution, which also saw improved systems of transportation, communication and banking. 

While industrialization brought about an increased 
volume and variety of manufactured goods and an improved standard of living for some, 
it also resulted in often grim employment and living conditions 
for the poor and working classes."""

parser = PlaintextParser.from_string(text, Tokenizer("english"))

for name, algo in [
    ("Default", LexRankSummarizer()),
    ("Creative", LuhnSummarizer()),
    ("Structured", LsaSummarizer()),
    ("Diverse", TextRankSummarizer())
]:
    print(f"\n{name}:")
    print(" ".join(map(str, algo(parser.document, 2))))
