python-kaggle-hashtag
=====================

My solution in Python for the kaggle problem Partly Sunny with a chance of Hashtag
I started Kaggle in december end, and that time no competition was running which interested me relatively.
So, although this competition had ended but due to the complexity and techniques used in this project, I chose to do this.
Here's the project link: http://www.kaggle.com/c/crowdflower-weather-twitter
Also,Since this was one of my first attempts I lacked some key techniques back then. But nevertheless it provided decent results, if you put the submission file.


Approach
---------
Since the problem dealt with keywords and phrases, we had to use NLP methods - owing to the brilliance of sklearn package, we could use the TFID techniques to
vectorize our input. The output was multi-output plus in 3 categories s,w,k. Hence used Ridge regression but with s,w,k broken into separate models. 
Since a problem of NLP at its best, The TFID approach was tuned with different parameters and customized tokenizers as well and error was checked. 
GridSearch was used to search for the best matching parameters as well.

Files
-------
hashtag.py - The core file containing the implementation
hash5-61.csv - Since I experimented a lot in this code, this was one of the setting when I got the best rank of 61. This is the submission file prepared from code.
hashtag-58 - The screenshot showing when I got rank 58 (total 259 teams were there) for one of the settings in my code.

Libraries used (Dependencies)
----------------
numpy (For arrays mostly)
pandas
sklearn (scikit-learn)





