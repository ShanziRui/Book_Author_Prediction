# Book_Author_Prediction
Use digital books from Gutenberg Library to train natural language processing models


The Operation Flow
===================


Setting up
------------

Download the submission folder on BrightSpace which contains 1 report document, 1 README file and 3 folders.

- Open the README file
- Unzip the "python_files" folder which contains 7 plain txt files and 2 python files.
- Open terminal, locate the directory of the submission folder, assume the directory is on your Desktop.
	cd /Users/catherinerui/Desktop/python_files
- Make sure "python_files" contains both "load.py" and "trainV4.py"
- Run trainV4.py using Python3
	python3 trainV4.py



Output Results
-----------------

The whole program takes a couple minutes to finish, here is a list of file output you will get.

- 7 folders, named by the corresponding book title, each contains the separated txt file of all chapters in that book.
	e.g. "pride-and-prejudice" contains 61 chapters in total, the program creates 61 txt files for each chapter, stores in the folder named "pride-and-prejudice-chapters"

- A visualization figure of population_size named "population_size.png"

- A visualization figure of the distribution of number of words in each document named "no_words_distribution.png"

- A visualization figure of word frequency obtained from bag of word after random sampling named "word_frequency.png"

- A visualization figure of box-plot for accuracy obtained by cross validation for each algorithm using BOW as input features named "bow_cross_validation.png"

- A visualization figure of confusion matrix named "confusion_matrix.png"




Program prints
-----------------

The output information printed in order:



************ Top 7 most frequently downloaded books ************

A Tale of Two Cities: Charles Dickens
The Adventures of Sherlock Holmes: Arthur Conan Doyle
Pride and Prejudice: Jane Austen
Frankenstein: Mary Wollstonecraft (Godwin) Shelley
Little Women: Louisa May Alcott
Moby Dick; or The Whale: Herman Melville
Adventures of Huckleberry Finn: Mark Twain (Samuel Clemens)




************ Population size of each book ************
A Tale of Two Cities 454
The Adventures of Sherlock Holmes 355
Pride and Prejudice 376
Frankenstein 251
Little Women 643
Moby Dick; or The Whale 724
Adventures of Huckleberry Finn 362



************ 1400 documents after sampling ************

                           Author                           Title                                           Document
                  Charles Dickens            A Tale of Two Cities  postpon take he gave rouleau gold i took hand ...
                  Charles Dickens            A Tale of Two Cities  say actual done prison made attempt life late ...
                  Charles Dickens            A Tale of Two Cities  wish see i grow unequ task i set it cold dark ...
                  Charles Dickens            A Tale of Two Cities  environ old chateau keep solitari state apart ...
                  Charles Dickens            A Tale of Two Cities  air madam defarg work hand accustom pass place...
                              ...                             ...                                                ...
      Mark Twain (Samuel Clemens)  Adventures of Huckleberry Finn  say i reckon bodi up tell truth tight place ta...
      Mark Twain (Samuel Clemens)  Adventures of Huckleberry Finn  chapter iii well i got good go morn old miss w...
      Mark Twain (Samuel Clemens)  Adventures of Huckleberry Finn  done what whole thing whi whole thing there on...
      Mark Twain (Samuel Clemens)  Adventures of Huckleberry Finn  said print offic we found littl bit concern ca...
      Mark Twain (Samuel Clemens)  Adventures of Huckleberry Finn  begun cuss town everybodi but duke say you bet...


[1400 rows x 3 columns]




************ Sampling size of each book ************
Author                                 Title  Document                                             
Arthur Conan Doyle                      200       200
Charles Dickens                         200       200
Herman Melville                         200       200
Jane Austen                             200       200
Louisa May Alcott                       200       200
Mark Twain (Samuel Clemens)             200       200
Mary Wollstonecraft (Godwin) Shelley    200       200


5.

************ Result for Support Vector Machine ************

The following part contains the accuracy indexed for 10-fold cross validation



The accuracy for Fold 1 : 0.9714285714285714
                  	Author          precision    recall  f1-score   support

                  Arthur Conan Doyle       0.95      1.00      0.98        20
                     Charles Dickens       1.00      0.90      0.95        20
                     Herman Melville       1.00      0.95      0.97        20
                         Jane Austen       0.95      1.00      0.98        20
                   Louisa May Alcott       1.00      0.95      0.97        20
         Mark Twain (Samuel Clemens)       0.95      1.00      0.98        20
	 Mary Wollstonecraft Shelley       0.95      1.00      0.98        20

                            accuracy                           0.97       140
                           macro avg       0.97      0.97      0.97       140
                        weighted avg       0.97      0.97      0.97       140

...



6.

************ Result for K-Nearest Neighbor ************

The same format with the above algorithm



7.

************ Result for MLP ************

The same format with the above algorithm



8.

************ Result for Decision Tree ************

The same format with the above algorithm

(each fold of cross validation will pop up a decision tree visualization figure, close the plt window after it pop up)


9.


************ Error Analysis ************

print out error records according to confusion matrix, where the number of wrong records are greater than 15

The output format is:
	'Correct author name' predicted as 'Predicted author name' : # examples.

e.g.

'Arthur Conan Doyle' predicted as 'Louisa May Alcott' : 11 examples.
Index   Author              Document
310  	Arthur Conan Doyle  flora millar confeder doubt respons disappear ...
202  Arthur Conan Doyle  men known coloni unnatur came settl near possi...
215  Arthur Conan Doyle  wall wood floor consist larg iron trough i cam...
376  Arthur Conan Doyle  jostl within space squar mile amid action reac...
224  Arthur Conan Doyle  much surpris interest glanc observ though boot...
253  Arthur Conan Doyle  parsonag cri rais sleeper bed it struck cold h...
308  Arthur Conan Doyle  ceil i think watson remark last case none fant...
295  Arthur Conan Doyle  met fresno street number constabl inspector wa...
361  Arthur Conan Doyle  therefor deceas gone bed within time deduct gr...
312  Arthur Conan Doyle  metal light set sun suffici mark hous even mis...
363  Arthur Conan Doyle  holm communic long drive lay back cab hum tune...
211  Arthur Conan Doyle  sudden door open end passag long golden bar li...
241  Arthur Conan Doyle  bride and onli one littl item anoth morn paper...
326  Arthur Conan Doyle  polic appear it certain sound feasibl well tak...
322  Arthur Conan Doyle  i ask want sum i ask next monday i larg sum du...
302  Arthur Conan Doyle  enter might privat word us this mr jabez wilso...
270  Arthur Conan Doyle  it dread think dear arthur prison i shall neve...
315  Arthur Conan Doyle  appear tell heavili home product one one manag...
233  Arthur Conan Doyle  understand this year good host windig name ins...
393  Arthur Conan Doyle  wrong i keep safeguard preserv weapon alway se...
342  Arthur Conan Doyle  thin wrinkl bent age opium pipe dangl knee tho...
260  Arthur Conan Doyle  tankervill club scandal ah cours he wrong accu...
279  Arthur Conan Doyle  write letter remain away i cannot imagin it un...
227  Arthur Conan Doyle  holder set streatham togeth devot hour glanc l...
247  Arthur Conan Doyle  caus robberi perpetr hous beat nativ butler de...
317  Arthur Conan Doyle  i fear much how done he spoke calm i could see...
399  Arthur Conan Doyle  shall busi afternoon shall probabl return lond...
204  Arthur Conan Doyle  tall man scotch bonnet coat button chin wait o...
303  Arthur Conan Doyle  north south east west everi man shade red hair...
299  Arthur Conan Doyle  jet ornament her dress brown rather darker cof...
319  Arthur Conan Doyle  claw crab thrown side in one wing window broke...
298  Arthur Conan Doyle  trivial anoth even other i right look help adv...
251  Arthur Conan Doyle  and see dead urgenc new case i urg young opens...
362  Arthur Conan Doyle  visibl upon wooden floor bedroom thrust away b...
240  Arthur Conan Doyle  this shall find but twelv mile drive gasp hath...
300  Arthur Conan Doyle  count good deal mysteri red head leagu i sure ...
276  Arthur Conan Doyle  set matter asid night bring explan it quarter ...
269  Arthur Conan Doyle  sir but mr fowler persev man good seaman block...
283  Arthur Conan Doyle  treasur near correct offici forc oh say mr jon...
338  Arthur Conan Doyle  worn blue dress seen man road pray continu sai...
320  Arthur Conan Doyle  prison turn reckless air man abandon destini b...
390  Arthur Conan Doyle  wilson yes i got answer thick red finger plant...
328  Arthur Conan Doyle  benefactor i learn gave hatherley farm rent fr...

