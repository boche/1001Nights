import RAKE

Rake = RAKE.Rake(RAKE.SmartStopList())
text = 'justices say capital cases must weigh war trauma'
text = 'news and notes about science'
result = Rake.run(text)
print(result)