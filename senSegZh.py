#! /bin/python
#-*- coding:utf-8 -*-
# how to run-> python senSegZh.py input >output

import re
import sys
import codecs

def readText(text):
    sentences = []
    for line in text:
	line = line.strip()
	line = sub(line)
	if line != '':
	    sentences.append(line)
    return sentences
		                                                                                                             
def sub(text):
    rep = {"/n+": "", "/s+": "","/t+":""} # define desired replacements here
    # use these three lines to do the replacement
    rep = dict((re.escape(k), v) for k, v in rep.iteritems())
    pattern = re.compile("|".join(rep.keys()))
    text = pattern.sub(lambda m: rep[re.escape(m.group(0))], text)
    return text
	
def splitText(sen):
    for i in sen:	
		split = re.split('(！！！|……|？？？|。”|？”|！”|？！|！？|。|？|！|;)',i)
		for r in range(len(split)):
			if r+1< len(split) and (r+1)%2!=0:
				print split[r],split[r+1]
				
if __name__=="__main__":		
    f = codecs.open(sys.argv[1])
    splitText(readText(f))



		



			
