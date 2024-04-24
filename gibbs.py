import numpy as np 
np.random.seed(0)
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from data import full_pages,vocab

def init(docs,vocab):
	global M,nt,N,n_words,words,alpha,eta,n_tw,n_dt
	M=len(docs)
	nt=10
	N=0
	words=[]
	n_words=len(vocab)
	for i in range(M):
		N=max(N,len(docs[i]))
		# n_words+=len(docs[i])
		words.append(len(docs[i]))
	alpha = np.random.gamma(shape=100, scale=0.01, size=1)  # one for all k
	eta = np.random.gamma(shape=100, scale=0.01, size=1)  # one for all V
	n_tw=np.zeros((nt,n_words),dtype=int)
	n_dt=np.zeros((M,nt),dtype=int)
  
def probs(word,d):
	prob=np.empty(nt)
	for i in range(nt):
		first=(n_tw[i,word]+eta)/(n_tw[i,:].sum()+eta*n_words)
		second=(n_dt[d,i]+alpha)/(n_dt[d,:].sum()+nt*alpha)
		prob[i]=first*second
	prob=prob/prob.sum()
	return prob

def gibbs(docs,n_iter,vocab):
	init(docs,vocab)
	global record
	record =np.zeros((M,N,n_iter+1),dtype=int)
	#initialisation of record array that is giviing topics to word randomly
	for i in range (M):
		for j in range (words[i]):
			word=docs[i][j]
			record[i,j,0]=np.random.randint(0,nt-1)
			x=record[i,j,0]
			n_tw[x,word]+=1
			n_dt[i,x]+=1
	for x in range (n_iter):
		for i in range (M):
			for j in range(words[i]):
				word=docs[i][j]
				prob=probs(word,i)
				old_topic=record[i,j,x]
				new_topic=np.argmax(np.random.multinomial(1,prob))
				record[i,j,x]=new_topic
				n_tw[old_topic,word]-=1
				n_dt[i,old_topic]-=1
				n_tw[new_topic,word]+=1
				n_dt[i,new_topic]+=1
				record[i,j,x+1]=new_topic
		# if (x+1)%10 == 0:
		print(x+1, 'iterations complete')
	return n_tw,n_dt
  
def heat(x_array,y_array,prob,doc):
	plt.figure(figsize=(8,8))
	plt.subplot(121)
	n_plot_words = 150
	sns.heatmap(prob.T[:n_plot_words], xticklabels=[], yticklabels=[])
	plt.xlabel("Topics", fontsize=14)
	plt.ylabel(f"Words[:{n_plot_words}]", fontsize=14)
	plt.title("topic-word distribution", fontsize=16)
	plt.subplot(122)
	sns.heatmap(doc, xticklabels=[], yticklabels=[])
	plt.xlabel("Topics", fontsize=14)
	plt.ylabel("Documents", fontsize=14)
	plt.title("document-topic distribution", fontsize=16)
	plt.tight_layout()
	plt.show()

def graph(docs,vocab):
	a,b=gibbs(docs,100,vocab)
	word_topic=np.zeros((nt,n_words),dtype=float)
	doc_topic=np.zeros((M,nt),dtype=float)
	# doc_topic=[]
	topic_array=np.arange(10)
	words_array=np.arange(n_words)
	for j in range (n_words):
		for i in range(nt):
			word_topic[i,j]=(a[i,j]+eta)/(a[i,:].sum()+n_words*eta)
	for i in range (M):
		for j in range (nt):
			doc_topic[i,j]=(b[i,j]+alpha)/(b[i,:].sum()+nt*alpha)
	heat(topic_array,words_array,word_topic,doc_topic)
  
docs=[]

for i in range (len(full_pages)):
		doc=[]
		for j in range (len(full_pages[i])):
			word=full_pages[i][j]
			doc.append(vocab.index(word))
		docs.append(doc)
	
graph(docs,vocab)
