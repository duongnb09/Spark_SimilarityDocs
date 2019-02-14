from pyspark import SparkConf, SparkContext
from operator import add
import os
import nltk
from nltk.stem import PorterStemmer
import time

#-------Set Input Parameters ------ #
inputPath = "/cosc6339_hw2/large-dataset/"
outputPath_Part1 = "top1kList.txt"
outputPath_Part2 = "/bigd16/Ass2_Part2.out"
outputPath_Part3 = "/bigd16/Ass2_Part3.out"
outputPath_Part4 = "top10SimilarDoc.txt"

nPartitions = 16 # Number of partitions
isSaveHDFSFile = False 

#-------nltk Parameters ------------#
# Use PorterStemmer for stemming task
ps = PorterStemmer()

# and download the stop word list if it does not exist
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = list(stopwords.words('english'))

# Pre-defined special characters
special_characters="0123456789~!@#$%^&*()+={}[]|\\;:\",<.>?/\n\r\t"

#--------Initialize spark configuration and context ----------#
conf = SparkConf()
conf.setAppName("PySpark_SimilarityDoc")
conf.set("spark.dynamicAllocation.enabled", "false")
conf.set("spark.shuffle.service.enabled", "false")
conf.set("spark.io.compression.codec", "snappy")
conf.set("spark.rdd.compress", "true")
conf.set("spark.executor.memory", "2g")
#conf.set("spark.sql.shuffle.partitions", "8")
sc = SparkContext(conf=conf)

#--------Supplemental functions for the first part-------------#
# Preprocess input text: 
# 1. Convert texts to lowercase 
# 2. Remove special characters
def preprocessText(text):
    # Step 1: Convert texts to lower case
    text = text.encode('ascii', 'ignore')
    text = text.lower(); 
    
    # Step 2: Remove special characters
    for i in range(0,len(special_characters)):
        text=text.replace(special_characters[i]," ")
    text = text.replace("'st","")    
    text = text.replace("'d","")
    text = text.replace("'s","")
    text = text.replace("'ll","")
    text = text.replace("--"," ")
        
    return text
    

# Filter stop or short words    
def filterWord(word):
    if (len(word)>1):
        for i in range(0,len(stop_words)):
            if (word==stop_words[i]):
                return False
        return True
    else:
        return False;
        
#--------Part 1 - Find the 1000 most popular words in the document collection ----# 
def find1KWords():
    print("Start the first part -  Find the 1000 most popular words")
    start_time = time.time()
    print("Part 1: Start reading input data (use sc.textFile func)")
    originalText=sc.textFile(inputPath)
   
    print("Part 1 - Step 1: Finish reading input data. Next, preprocess texts (use map func)")
    preprocessedText = originalText.map(lambda line:preprocessText(line)) # Preprocess data
    
    print("Part 1 - Step 2: Finish preprocessing data! Next, split words (use flatMap function)")
    words = preprocessedText.flatMap(lambda line:line.split()) # Extract words
    
    print("Part 1 - Step 3: Finish splitting words! Next, stem words (use map func)")
    stemmedWords = words.map(lambda word:ps.stem(word)) # Stem words
    
    print("Part 1 - Step 4: Finish stemming words! Next, filter stop words and short words (use filter func")
    finalWords = stemmedWords.filter(lambda word: filterWord(word)) # Filter stop or short words
   
    print("Part 1 - Step 5: Finish preprocessing phrase. The final set of words is generated. Next, count the words (use map func)")
    wordCount = finalWords.map(lambda word:(word,1))
   
    print("Part 1 - Step 6: Finish word mapping! Next, perform reducing task (use reduceByKey func)")
    counts = wordCount.reduceByKey(add)
   
    print("Part 1 - Step 7: Finish the reducint task! Next, find top 1000 words (use takeOrdered func). It will take quite a long time...")
    top1kList = counts.takeOrdered(1000, key = lambda item:-item[1])
    
    
    print("Part 1: The top 1k words are extracted. Finish the first part. ")
    print("---Running time of the first part: %s seconds ---" % (time.time() - start_time))
    print("Wait for a couple of seconds before starting the second part. The program will save the result to a text file.")
    top1kWords = []
    with open(outputPath_Part1, 'w') as f:
        for item in top1kList:
            f.write( str(item[0])+","+str(item[1])+"\n")
            top1kWords.append(item[0])
    print("Done saving file for the firt part")
    print("Top 1k words:")
    print(top1kList)
    return top1kWords
 
#--------Supplemental functions for the second part-------------#

# Split text to invididual words, stem words and filter stop words
# Input: Key-value pair with the key is a document name and the value is the text content
# Output: A list of key-value pairs with the key is a document name, the value is a word
def splitText(item):
    outputList=[]
    textData = item[1].encode('ascii', 'ignore')
    docWords = textData.split()
    for w in docWords:
        stemmedWord = ps.stem(w)  
        if (stemmedWord not in stop_words) and (len(stemmedWord)>0):
            outputList.append((item[0],stemmedWord))
    return outputList

# Read words from the inputFile 
def readTop1KWords(inputFile):
    top1kList=[]
    with open(inputFile, 'r') as f: 
        for value in f:
            v = value.replace('\n','')
            v = v.replace('\r','')
            top1kList.append(v)
    top1kSet = set(top1kList)
    return top1kSet

#Calculate a weight for a word in a document
#Input: A key-value pair with the key is the document name, the value is a list of (word,number of occurences) pairs
#Output: A list of key-value pairs with the key is the document name, and the value is (word,weight) pair.
def calculateWeight(item):
    inputWordList = item[1]
    fileName = item[0]
    values = []
    totalNumbWords = 0
    for v in inputWordList:
        totalNumbWords = totalNumbWords + int(v[1])
    for v in inputWordList:
        weight = float(v[1])/totalNumbWords
        singleItem = (fileName,(v[0],weight))
        values.append(singleItem)
    return values
    
# Unwrap the output from the groupByKey function
# Input: A key-value pair with the key is a word, 
#        and the value is an iterable sequence object which is not readable by humans: 
#        IterableSeq((doc1,weight1),(doc2,weight2)...)
# Output: A key-value pair with the key is a word, and a list of (doc,weight) pairs which are readble by humans 
def unwrapIterableSeq(item):
    output = []
    for v in item[1]:
        output.append(v)
    return (item[0],output)

#The three following function are used for the combineByKey function    
def to_list(a):
    return [a]

def append(a,b):
    a.append(b)
    return a

def extend(a,b):
    a.extend(b)
    return a
        
#--------Part 2 - Create an inverted index for the top 1,000 words  ----#
def createInvertedIndex():
    #top1kSet = readTop1KWords(outputPath_Part1) # For testing
    
    top1kSet = set(find1KWords()) # The reason for using set: searching in a set is faster in a list
    
    print("Start the second part -  Create an inverted index for the top 1k words")
    start_time = time.time()
    print("Part 2 - Step 1: First, read the input data (use wholeTextFiles func)")
    textWithPath = sc.wholeTextFiles(inputPath).repartition(nPartitions)
    
    print("Part 2 - Step 2: Finish reading input data. Next, preprocess texts (use map func)")
    preprocessedTextWithPath = textWithPath.map(lambda item:(os.path.basename(item[0]),preprocessText(item[1])))

    print("Part 2 - Step 3: Finish preprocessing texts. Next, split texts to individual words (use flatMap func)")
    wordsWithPath = preprocessedTextWithPath.flatMap(lambda item:splitText(item))
    
    print("Part 2 - Step 4: Finish splitting words. Next, select words in the top 1k list (use filter func)")
    final1KWordswithPath = wordsWithPath.filter(lambda item: item[1] in top1kSet)
    
    print("Part 2 - Step 5: Finish filtering words. Next, perform word count mapping task (use map func)")
    wordCountMapper = final1KWordswithPath.map(lambda item:((item[0],item[1]),1))
    
    print("Part 2 - Step 6: Finish wordcount mapping. Next, perform the reducing task (use reduceByKey func")
    wordCountReducer = wordCountMapper.reduceByKey(add,numPartitions=nPartitions)
    
    print("Part 2 - Step 7: Finish wordcount reducing. Next, change key-value representation (use map func)")
    document_wordweight_pair = wordCountReducer.map(lambda item:(item[0][0],(item[0][1],item[1])))
    
    print("Part 2 - Step 8: Finish chaning key-value representation. Next, group words for each document (use combineByKey or groupByKey func)")
    groupWords = document_wordweight_pair.combineByKey(to_list,append,extend)
    
    print("Part 2 - Step 9: Finish grouping word. Next, compute weights for words in each document ( use flatMap)")
    documentWeight = groupWords.flatMap(lambda item:calculateWeight(item))
    
    print("Part 2 - Step 10: Finish computing weight. Next, change key-value representation, make words as keys (use map func)")
    wordAsKeys = documentWeight.map(lambda item:(item[1][0],(item[0],item[1][1])))
    
    print("Part 2 - Step 11: Finish chaning key-value representation. Group results to get the final inverted index (use combineByKey or groupByKey func)")
    invertedIndexRDD = wordAsKeys.combineByKey(to_list,append,extend)
    
    print("Part 2 - Step 12: Perform the collect action to get all results. Please wait...")
    
    invertedIndex = invertedIndexRDD.collect()
    
    
    print("---Part 2: Running time: %s seconds ---" % (time.time() - start_time))
    
    
    #readableInvertedIndex = invertedIndex.map(lambda item:unwrapIterableSeq(item))
    #readableInvertedIndex.collect()
    if isSaveHDFSFile:
        print("Part 2 - The inverted index is created. Now, the results will be saved to files. This is the most time consuming task...")
        invertedIndexRDD.coalesce(1).saveAsTextFile(outputPath_Part2)
    
    print("Part 2 - Finish")
    return invertedIndex


#------ Supplemental functions for the third part ------- #

# Create doc pairs from inverted index
# Input: Inverted index with a key is a word and the value is list of (doc,weight) pair
# Output: A list of ((docx,docy), combined weight)
# item[0] = word, item[1] = list((doc1,weight1),(doc2,weight2)...) 
def createDocPairs(item):
    n = len(item[1])
    outputList=[]
    for docx in item[1]:
        for docy in item[1]:
            namex = docx[0]
            weightx = docx[1]
            
            namey = docy[0]
            weighty = docy[1]
            combinedWeight = weightx*weighty
            if (namex!=namey):
                if (namex<namey):
                    value = ((namex,namey),combinedWeight)
                    outputList.append(value)
                else:
                    value = ((namey,namex),combinedWeight)
                    outputList.append(value)
                
    return outputList
    
#--------Part 3 - Compute similarity matrix  ----#

def computeSimilarityMatrix():
    invertedIndex = sc.parallelize(createInvertedIndex())

    print("Start the third and fourth part - Compute similarity matrix")
    start_time = time.time() 
    similarityMatrix = invertedIndex.flatMap(lambda item:createDocPairs(item)).reduceByKey(add,numPartitions=8)
    if isSaveHDFSFile:
        similarityMatrix.coalesce(1).saveAsTextFile(outputPath_Part3)

    print("Part 4 - Find the top 10 similar document (use takeOrdered func). It'll take some time...")
    top10SimilarDoc = similarityMatrix.takeOrdered(10, key = lambda item:-item[1])

    #similarityMatrix.coalesce(1,False).saveAsTextFile(outputPath_Part3)
    #similarityMatrix.coalesce(1).saveAsTextFile(outputPath_Part3)
    print("---Part3&4: Running time: %s seconds ---" % (time.time() - start_time))

    with open(outputPath_Part4, 'w') as f:
        for docPair in top10SimilarDoc:
            f.write( docPair[0][0]+"-"+docPair[0][1]+", Similarity Value: "+str(docPair[1])+"\n")
    print("Finish saving file for the fourth part")
    print("Completely Done!")
   
if __name__ == "__main__":
    # Call the computeSimilarityMatrix function will trigger both functions for the first and second parts
    computeSimilarityMatrix()    





 
