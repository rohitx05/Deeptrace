import codecs
text = codecs.open('test_results.txt', 'r', 'utf-16le').read()
with codecs.open('metrics_clean.txt', 'w', 'utf-8') as f:
    f.write(text)
